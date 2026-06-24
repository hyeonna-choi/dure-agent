"""
natural_evaluator_llm.py — Claude API-based comparison of the [1]-[3] NATURAL sections.

Instead of rule-based parameter extraction, the LLM interprets the context and
compares [1]-[3] between GT and Pred, judging omissions and errors.

Core evaluation criteria:
- Only µL (volume), times (count), and minutes/hours (time) parameters are evaluated.
- Even the same parameter value is treated separately if it belongs to a different action.
- Omitting the re-mention of content already mentioned elsewhere is not an omission.
- Over-generation (information present only in Pred) is not counted as an error unless the content is incorrect.
"""

import json
import os
from typing import Dict, List, Tuple

from anthropic import Anthropic
from .metrics import AccuracyMetrics, CompletenessMetrics


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
You are an expert evaluator of ELISA protocols.

Compare the [1]-[3] NATURAL sections between GT (ground truth) and Pred
(prediction), and identify important parameter information in Pred that is
**actually omitted or incorrect**.

## Parameters to evaluate (evaluate only these)
- µL : volume (dispensing amount, wash amount, etc.)
- times : count (wash count, etc.)
- minutes/hours : time (incubation time, etc.)

## What not to evaluate (ignore)
- Concentration (µg/mL, pg/mL, %)
- Temperature (°C)
- Dilution factor (X, fold)
- Kit reconstitution details (vials, reconstitution volume, etc.)
- Manual information such as storage conditions and filtration methods

## Key rules
1. **Even the same parameter value is separate if it belongs to a different action**: "coating 100µL" and "sample dispensing 100µL" are different items.
2. **Omitting a re-mention is not an omission**: if GT repeats the same content multiple times but Pred wrote it only once, it is not an omission because it was already mentioned.
3. **Mentioning it in another section is also OK**: if content from GT [2] appears in Pred [1], it is not an omission.
4. **Over-generation is not an error**: additional information present only in Pred is ignored unless the content is incorrect.
5. **Only differing values are errors**: GT "incubation 1 hour" vs Pred "incubation 30 minutes" -> values differ -> error.
6. **Solution removal (aspirate) volume is excluded from evaluation**: solution removal aspirates all of the solution in the well, so omissions or differences in the solution removal volume are not counted as errors.

## Output format (you must use exactly this JSON format)
```json
{
  "total_items": <number of unique actions in GT that contain an evaluated parameter>,
  "correct_items": <number of actions correctly covered in Pred>,
  "errors": [
    {
      "type": "missing" | "wrong_value",
      "gt_line": "<GT source line>",
      "pred_line": "<corresponding Pred line or null>",
      "param": "<description of the parameter in question>",
      "reason": "<brief reason>"
    }
  ]
}
```

If there are no errors, output an empty array [].\
"""


_USER_TEMPLATE = """\
Compare the [1]-[3] sections of GT and Pred below.

=== GT [1]-[3] ===
{gt_text}

=== Pred [1]-[3] ===
{pred_text}

Respond only in the JSON format specified above.\
"""


def evaluate_natural_with_llm(
    gt_natural: str,
    pred_natural: str,
    model: str = "claude-sonnet-4-6",
) -> Tuple[CompletenessMetrics, AccuracyMetrics, List[dict]]:
    """
    Compare the [1]-[3] NATURAL sections using the LLM.

    Returns:
        CompletenessMetrics: TP/FP/FN -> F1
        AccuracyMetrics: correct/total
        details: [{section, errors: [...], missing_from_pred, missing_source_lines}]
    """
    from .line_parser import parse_natural_sections

    gt_secs = parse_natural_sections(gt_natural)
    pred_secs = parse_natural_sections(pred_natural)

    # Extract the [1]-[3] text
    gt_lines = []
    pred_lines = []
    for sec_num in ("1", "2", "3"):
        gt_sec_lines = gt_secs.get(sec_num, [])
        pred_sec_lines = pred_secs.get(sec_num, [])
        if gt_sec_lines:
            gt_lines.append(f"[{sec_num}]")
            gt_lines.extend(gt_sec_lines)
        if pred_sec_lines:
            pred_lines.append(f"[{sec_num}]")
            pred_lines.extend(pred_sec_lines)

    gt_text = "\n".join(gt_lines).strip()
    pred_text = "\n".join(pred_lines).strip()

    # If both are empty -> perfect match
    if not gt_text and not pred_text:
        return CompletenessMetrics(), AccuracyMetrics(), []

    # If GT is empty -> comparison impossible, 0/0
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
        print(f"    [LLM Natural] JSON parsing error: {e}", flush=True)
        print(f"    [LLM Natural] Raw Claude response (first 300 chars):\n      {_raw_response[:300]}", flush=True)
        raise
    except Exception as e:
        print(f"    [LLM Natural] API error: {type(e).__name__}: {e}", flush=True)
        raise

    # Parse the result
    total = result.get("total_items", 0)
    correct = result.get("correct_items", 0)
    errors = result.get("errors", [])

    # Completeness: TP=correct, FN=total-correct; FP is hard for the LLM to judge directly, so 0
    comp = CompletenessMetrics()
    comp.TP = correct
    comp.FP = 0
    comp.FN = total - correct

    acc = AccuracyMetrics()
    acc.total = total
    acc.correct = correct

    # Build details
    details = []
    if errors:
        detail_entry = {"section": "[1]-[3] LLM evaluation"}

        missing_list = []
        source_lines = []
        for err in errors:
            err_type = err.get("type", "")
            gt_line = err.get("gt_line", "")
            pred_line = err.get("pred_line", "")
            param = err.get("param", "")
            reason = err.get("reason", "")

            if err_type == "missing":
                missing_list.append(f"{param} — {reason}")
            elif err_type == "wrong_value":
                missing_list.append(f"{param}: {reason}")

            if gt_line:
                source_lines.append(gt_line)

        if missing_list:
            detail_entry["missing_from_pred"] = missing_list
        if source_lines:
            detail_entry["missing_source_lines"] = list(dict.fromkeys(source_lines))

        detail_entry["llm_errors"] = errors
        details.append(detail_entry)

    return comp, acc, details
