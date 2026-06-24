"""
instrument_evaluator_llm.py — Claude API-based comparison of the [5] INSTRUMENT section.

In addition to the rule-based approach (LCS + parameter comparison), the LLM
interprets the context and compares the [5] automation execution section between
GT and Pred, judging omissions, over-generation, and parameter errors.

Core evaluation criteria:
- Compare each action unit such as incubation, solution removal, washing,
  dispensing, shaking, and reading.
- Parameters: µL (volume), seconds/minutes/hours (time), times (wash count),
  nm (wavelength), rpm, reservoir number.
- Both the execution order and the parameter values must be accurate.
- Over-generated actions present only in Pred are also evaluated as errors.
"""

import json
import os
from typing import Dict, List, Tuple

from anthropic import Anthropic
from .metrics import CompletenessMetrics, AccuracyMetrics


_CLIENT = None

def _get_client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        import os
        api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY environment variable is not set.")
        _CLIENT = Anthropic(api_key=api_key)
    return _CLIENT


_SYSTEM_PROMPT = """\
You are an expert evaluator of ELISA protocol automation.

Compare the [5] automation execution (INSTRUMENT) section between GT (ground
truth) and Pred (prediction), and identify actions in Pred that are **omitted,
over-generated, or have incorrect parameters**.

## Action types
- Incubation: time (seconds/minutes/hours), temperature
- Solution removal: volume (µL)
- Washing: reservoir number, count (times), volume (µL)
- Dispensing: reservoir number, volume (µL)
- shaking: time (seconds/minutes), RPM
- Reading: wavelength (nm)

## Parameters to compare (compare only these)
Compare only the following numeric parameters:
- **Volume** (µL): e.g., 100µL, 300µL
- **Time** (minutes/hours/seconds): e.g., 30 minutes, 2 hours, 1800 seconds
- **Count** (times): e.g., 4 times, 3 times
- **Reservoir number**: e.g., Reservoir 1, R2 (number only)
- **RPM**: e.g., 500rpm
- **Wavelength** (nm): e.g., 450nm
- **Temperature** (°C): e.g., 37°C, 20°C

## Items to never compare (ignore these)
Do not treat the following as errors:
- **Reagent/solution names**: e.g., "Streptavidin-HRP" vs "Streptavidin-HRP Conjugate", "TMB Substrate" vs "TMB Substrate Solution", "Stop Solution" vs "1M HCl Stop Solution" -> reagent names are evaluated separately in reservoir mapping, so ignore them here
- **Concentration** (µg/mL, pg/mL, %): e.g., 0.25 µg/mL -> this is kit manual information, so ignore it
- **Time unit differences**: "120 minutes" = "2 hours" = "7200 seconds" is the same value -> normalize units before comparing
- **Qualifiers such as "or more", "at least"**: "2 hours or more" = "2 hours" -> compare only the numeric value
- **Punctuation differences such as periods and commas**
- **Incubation location/condition phrasing**: e.g., "room temperature", "benchtop", "shaker"
- **"light-blocked", "light-shielded", "light-protected"**: same meaning -> ignore
- **Wording differences**: different wording with the same meaning is not an error

## Reservoir number differences — mostly allowed
- Wash Buffer may be located at any reservoir position.
- If the Wash Buffer position changes, **the reservoir numbers of all other reagents also shift in a chained manner** -> this is **entirely normal**.
- **Decision rule**: if the action type and order are the same between GT and Pred, ignore all reservoir number differences.
  - Example: GT: wash R14 -> dispense R15 (DetAb) -> wash R14 -> dispense R16 (SA-HRP) -> wash R14
  - Pred: wash R16 -> dispense R14 (DetAb) -> wash R16 -> dispense R15 (SA-HRP) -> wash R16
  - -> **All 5 actions are normal**. The action types and order are identical; only the reservoir numbers differ.
- **Report an error only when**: the action type itself differs, the action order differs, or a non-reservoir parameter such as volume/time/count differs.
- In other words, **a reservoir number difference alone is never an error**.

## Shaking parameters — allowed
- "Short shaking (mixing)" automatically applies the system defaults (200 rpm, 10 seconds).
- If rpm/time is absent in GT but present in Pred (or vice versa) -> normal, not an error.
- Never report rpm or time differences in shaking/tap as errors.

## Evaluation criteria
1. **Action matching**: whether each GT action exists in Pred in order (by action type).
2. **Parameter accuracy**: for matched action pairs, compare only the numeric values of the "parameters to compare" above.
   - **Reservoir number**: if the action type/order match, ignore all reservoir number differences (chained shifting due to Wash Buffer relocation is possible).
   - Time: normalize units (30 minutes = 1800 seconds = 0.5 hours) before comparing.
   - Volume, count, reservoir number, nm, rpm, temperature: numbers must match exactly.
3. **Omission (FN)**: actions present in GT but absent in Pred.
4. **Over-generation (FP)**: actions present only in Pred and absent in GT.

## Output format (you must use exactly this JSON format)
```json
{
  "gt_action_count": <number of GT actions>,
  "pred_action_count": <number of Pred actions>,
  "matched_count": <number of actions matched in order (TP)>,
  "fn_count": <number of actions present only in GT>,
  "fp_count": <number of actions present only in Pred>,
  "param_total": <number of comparable parameters across matched actions (numeric parameters only)>,
  "param_correct": <number of numeric parameters that match exactly>,
  "errors": [
    {
      "type": "missing" | "extra" | "wrong_param",
      "gt_line": "<GT source line or null>",
      "pred_line": "<Pred source line or null>",
      "action_desc": "<action description (e.g., incubation 1800s, wash R1 300µL 4 times)>",
      "param": "<incorrect parameter name (for wrong_param)>",
      "gt_val": "<GT parameter value (for wrong_param)>",
      "pred_val": "<Pred parameter value (for wrong_param)>",
      "reason": "<brief reason>"
    }
  ]
}
```

If there are no errors, output an empty array [].\
"""


_USER_TEMPLATE = """\
Compare the [5] automation execution sections of GT and Pred below.

=== GT [5] ===
{gt_text}

=== Pred [5] ===
{pred_text}

Respond only in the JSON format specified above.\
"""


def evaluate_instrument_with_llm(
    gt_instrument: str,
    pred_instrument: str,
    model: str = "claude-sonnet-4-6",
) -> Tuple[CompletenessMetrics, AccuracyMetrics, List[dict]]:
    """
    Compare the [5] INSTRUMENT section using the LLM.

    Returns:
        CompletenessMetrics: TP/FN/FP
        AccuracyMetrics: param correct/total
        details: [{errors: [...]}]
    """
    from .line_parser import extract_instrument_text, parse_section5

    # Extract the [5] text
    gt_lines = []
    pred_lines = []
    in_section = False

    for line in gt_instrument.splitlines():
        stripped = line.strip()
        if stripped.startswith("[5]"):
            in_section = True
            gt_lines.append(stripped)
            continue
        if stripped.startswith("[6]"):
            break
        if in_section and stripped:
            gt_lines.append(stripped)

    in_section = False
    for line in pred_instrument.splitlines():
        stripped = line.strip()
        if stripped.startswith("[5]"):
            in_section = True
            pred_lines.append(stripped)
            continue
        if stripped.startswith("[6]"):
            break
        if in_section and stripped:
            pred_lines.append(stripped)

    gt_text = "\n".join(gt_lines).strip()
    pred_text = "\n".join(pred_lines).strip()

    # If both are empty
    if not gt_text and not pred_text:
        return CompletenessMetrics(), AccuracyMetrics(), []

    if not gt_text:
        return CompletenessMetrics(), AccuracyMetrics(), []

    # Call the Claude API
    client = _get_client()
    user_msg = _USER_TEMPLATE.format(gt_text=gt_text, pred_text=pred_text)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=16384,
            temperature=0,
            timeout=120,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_msg},
            ],
        )

        result_text = response.content[0].text
        _raw_response = result_text[:500]

        # Extract the JSON block (including handling for truncated responses)
        import re as _re
        _json_block = _re.search(r'```json\s*(.*?)```', result_text, _re.S)
        if not _json_block:
            _json_block = _re.search(r'```\s*(.*?)```', result_text, _re.S)
        if not _json_block:
            _json_block = _re.search(r'```json\s*(.*)', result_text, _re.S)
        if not _json_block:
            _json_block = _re.search(r'```\s*(.*)', result_text, _re.S)
        if _json_block:
            result_text = _json_block.group(1).strip()

        result = json.loads(result_text)

    except json.JSONDecodeError as e:
        print(f"    [LLM Instrument] JSON parsing error: {e}", flush=True)
        print(f"    [LLM Instrument] Raw Claude response (first 300 chars):\n      {_raw_response[:300]}", flush=True)
        raise
    except Exception as e:
        print(f"    [LLM Instrument] API error: {type(e).__name__}: {e}", flush=True)
        raise

    # Parse the result
    comp = CompletenessMetrics(
        TP=result.get("matched_count", 0),
        FN=result.get("fn_count", 0),
        FP=result.get("fp_count", 0),
    )

    param_acc = AccuracyMetrics()
    param_acc.total = result.get("param_total", 0)
    param_acc.correct = result.get("param_correct", 0)

    errors = result.get("errors", [])

    # Build details
    details = []
    if errors:
        detail_entry = {"section": "[5] LLM evaluation", "llm_errors": errors}

        fn_list = []
        fp_list = []
        wrong_list = []
        for err in errors:
            err_type = err.get("type", "")
            action_desc = err.get("action_desc", "")
            reason = err.get("reason", "")

            if err_type == "missing":
                fn_list.append(f"{action_desc} — {reason}")
            elif err_type == "extra":
                fp_list.append(f"{action_desc} — {reason}")
            elif err_type == "wrong_param":
                param = err.get("param", "")
                gt_val = err.get("gt_val", "")
                pred_val = err.get("pred_val", "")
                wrong_list.append(f"{action_desc} → {param}: GT={gt_val} Pred={pred_val}")

        if fn_list:
            detail_entry["fn_actions"] = fn_list
        if fp_list:
            detail_entry["fp_actions"] = fp_list
        if wrong_list:
            detail_entry["wrong_params"] = wrong_list

        details.append(detail_entry)

    return comp, param_acc, details
