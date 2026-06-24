"""
reservoir_evaluator_llm.py — Claude API-based comparison of the [4] reservoir mapping.

Instead of the rule-based approach (text matching + synonyms), the LLM interprets
the meaning and compares the reservoir mapping between GT and Pred, judging
mismatches.

Core evaluation criteria:
- For each reservoir number, semantically compare whether the GT and Pred reagent
  names refer to the same reagent.
- Ignore formatting differences (parentheses, dashes, slashes, etc.).
- Treat a user-defined Calibrant (User-defined) as normal.
- Ignore differences in supplementary information such as concentration and volume,
  and compare only the reagent type.
"""

import json
from typing import Dict, List, Tuple

from anthropic import Anthropic
from .metrics import AccuracyMetrics


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

Compare the [4] reservoir mapping between GT (ground truth) and Pred (prediction),
and identify reservoirs in Pred where the **reagent is mapped incorrectly**.

## Comparison criteria
1. For the **same reservoir number**, compare whether the GT reagent and the Pred reagent are the **same type of reagent**.
2. The following are considered the **same reagent** (not errors):
   - Formatting/notation differences: "Standard S1 (10 pg/mL)" = "Standard/Calibrant S1 — 10 pg/mL"
   - Abbreviation differences: "Wash Buffer" = "WB" = "Wash"
   - Supplementary information differences: "TMB Substrate Solution" = "TMB Substrate" = "TMB"
   - "Stop Solution" = "1M HCl Stop Solution" = "HCl"
   - "Detection Antibody" = "Detection Ab"
   - "Streptavidin-HRP" = "SA-HRP" = "Streptavidin-HRP Conjugate"
   - "Assay Diluent" = "Diluent"
   - Concentration/volume differences: ignore numeric differences in concentration (µg/mL, pg/mL) or volume (µL)
3. **User-defined Calibrant (allowed in both directions)**:
   - A Calibrant in Pred that includes "user-defined" or "User-defined" is normal.
   - A Calibrant in GT that includes "user-defined" or "User-defined" while Pred has the actual Standard name/concentration is also **normal**.
   - Example: GT="user-defined Calibrant 2 (User-defined, concentration not specified)" vs Pred="Standard S2 (500 pg/mL)" -> **normal** (Pred found the actual value from the PDF).
   - In other words, if either GT or Pred is User-defined, treat that Calibrant as normal.
4. **Blank/Standard/Calibrant/Sample**: ignore wording differences in the type name (Standard = Calibrant).
5. Report reservoir numbers present only in GT or only in Pred as **omissions/over-generation**.

## Wash Buffer relocation fully allowed
- In actual automation, Wash Buffer may be located at any reservoir position.
- If Wash Buffer is at R14 in GT and at R16 in Pred, the Wash Buffer itself is **normal**.
- The shifting of other reagents' reservoir numbers due to Wash Buffer relocation is also **normal** (chained relocation allowed).
- Example: GT: R14=Wash, R15=DetAb, R16=SA-HRP -> Pred: R14=DetAb, R15=SA-HRP, R16=Wash
  -> All 3 are **normal**. Wash simply shifted back and the rest pulled forward.
- Decision method: if the reagent lists (types) of GT and Pred are identical, all number changes caused by Wash Buffer position differences are normal.
- However, report an error only when the reagent type itself differs or is omitted/over-generated.

## Key point: compare only the reagent "type"
- Wash Buffer, Assay Diluent, Detection Antibody, Streptavidin-HRP, TMB Substrate, Stop Solution, etc.
- Same type means a match; different type means a mismatch.
- Also compare the Standard/Calibrant number (S1, S2, etc.).

## Output format (you must use exactly this JSON format)
```json
{
  "total_reservoirs": <number of reservoirs to compare (present on both sides)>,
  "correct_count": <number of correctly mapped reagents>,
  "errors": [
    {
      "type": "mismatch" | "missing_in_pred" | "extra_in_pred",
      "reservoir": <reservoir number>,
      "gt_reagent": "<GT reagent name or null>",
      "pred_reagent": "<Pred reagent name or null>",
      "reason": "<brief reason>"
    }
  ]
}
```

If there are no errors, output an empty array [].\
"""


_USER_TEMPLATE = """\
Compare the [4] reservoir mapping of GT and Pred below.

=== GT [4] reservoir mapping ===
{gt_text}

=== Pred [4] reservoir mapping ===
{pred_text}

Respond only in the JSON format specified above.\
"""


def evaluate_reservoir_with_llm(
    gt_natural: str,
    pred_natural: str,
    model: str = "claude-sonnet-4-6",
) -> Tuple[AccuracyMetrics, List[dict]]:
    """
    Compare the [4] reservoir mapping using the LLM.

    Returns:
        AccuracyMetrics: correct/total
        details: [{"reservoir": ..., "gt_reagent": ..., "pred_reagent": ..., "match": bool}]
    """
    # Extract the [4] section text
    gt_lines = _extract_section4(gt_natural)
    pred_lines = _extract_section4(pred_natural)

    gt_text = "\n".join(gt_lines).strip()
    pred_text = "\n".join(pred_lines).strip()

    # If both are empty
    if not gt_text and not pred_text:
        return AccuracyMetrics(), []

    if not gt_text:
        return AccuracyMetrics(), []

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
        print(f"    [LLM Reservoir] JSON parsing error: {e}", flush=True)
        print(f"    [LLM Reservoir] Raw Claude response (first 300 chars):\n      {_raw_response[:300]}", flush=True)
        raise
    except Exception as e:
        print(f"    [LLM Reservoir] API error: {type(e).__name__}: {e}", flush=True)
        raise

    # Parse the result
    acc = AccuracyMetrics()
    acc.total = result.get("total_reservoirs", 0)
    acc.correct = result.get("correct_count", 0)

    errors = result.get("errors", [])

    # Build details — a format compatible with the existing rule-based output
    details = []
    for err in errors:
        err_type = err.get("type", "")
        reservoir = err.get("reservoir", 0)
        gt_reagent = err.get("gt_reagent") or "(missing)"
        pred_reagent = err.get("pred_reagent") or "(missing)"
        reason = err.get("reason", "")

        details.append({
            "reservoir": reservoir,
            "gt_reagent": gt_reagent,
            "pred_reagent": pred_reagent,
            "match": False,
            "error_type": err_type,
            "reason": reason,
        })

    return acc, details


def _extract_section4(text: str) -> List[str]:
    """Extract the [4] section text."""
    import re
    lines = []
    in_section = False

    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"\[4\]", stripped):
            in_section = True
            lines.append(stripped)
            continue
        if in_section and re.match(r"\[5\]", stripped):
            break
        if in_section and stripped.startswith("</"):
            break
        if in_section and stripped:
            lines.append(stripped)

    return lines
