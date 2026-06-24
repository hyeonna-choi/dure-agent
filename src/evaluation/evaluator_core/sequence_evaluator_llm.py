"""
sequence_evaluator_llm.py — Claude API-based comparison of sequence commands.

In addition to the rule-based approach (block parsing + LCS matching + parameter
comparison), the LLM compares the parsed block list as text and judges omissions,
over-generation, and parameter errors.

Block parsing is performed rule-based, and the parsed result is converted to text
and passed to the LLM.
"""

import json
import os
from typing import Dict, List, Tuple

import pandas as pd
from anthropic import Anthropic
from .block_parser import parse_blocks, SeqBlock
from .metrics import CompletenessMetrics, AccuracyMetrics, OrderMetrics


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


def _blocks_to_text(blocks: List[SeqBlock]) -> str:
    """Convert the block list into human-readable text."""
    lines = []
    for idx, b in enumerate(blocks, 1):
        parts = [f"#{idx} [{b.type}]"]
        if b.reservoir_col is not None:
            parts.append(f"Reservoir={b.reservoir_col}")
        if b.volume is not None:
            parts.append(f"Volume={b.volume}µL")
        if b.wait_seconds is not None:
            parts.append(f"Wait={b.wait_seconds}s")
        if b.temperature is not None:
            parts.append(f"Temp={b.temperature}°C")
        if b.wash_cycles is not None:
            parts.append(f"WashCycles={b.wash_cycles}")
        if b.rpm is not None:
            parts.append(f"RPM={b.rpm}")
        if b.shake_seconds is not None:
            parts.append(f"ShakeTime={b.shake_seconds}s")
        parts.append(f"(rows {b.start_idx}-{b.end_idx})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are an expert evaluator of ELISA protocol automation sequence commands.

Compare the sequence command block lists between GT (ground truth) and Pred
(prediction), and identify blocks in Pred that are **omitted, over-generated,
out of order, or have incorrect parameters**.

## Block types
- Initial setup: initial configuration at the start of the sequence
- Incubation: Wait (seconds), Temperature (°C)
- Solution removal: Volume (µL)
- Washing: reservoir number, Volume (µL), WashCycles (times)
- Dispensing: reservoir number, Volume (µL)
- shaking: RPM, ShakeTime (seconds)
- Reading: reader operation

## Reservoir number differences — mostly allowed
- Wash Buffer may be located at any reservoir position.
- If the Wash Buffer position changes, **the reservoir numbers of all other reagents also shift in a chained manner** -> this is **entirely normal**.
- **Decision rule**: if the block type and order are the same between GT and Pred, ignore all reservoir number differences.
  - Example: GT: wash R14 -> dispense R15 -> wash R14 -> dispense R16 -> wash R14
  - Pred: wash R16 -> dispense R14 -> wash R16 -> dispense R15 -> wash R16
  - -> **All 5 blocks are normal**. The block types and order are identical; only the reservoir numbers differ.
- **Report an error only when**: the block type itself differs, a block is omitted/over-generated, or a non-reservoir parameter such as volume/time/count differs.
- In other words, **a reservoir number difference alone is never an error**.

## Shaking parameters — allowed
- RPM and ShakeTime differences in shaking/tap blocks are not errors.
- The system automatically applies the defaults (200 rpm, 10 seconds), so the shaking parameters of GT and Pred may differ and still be normal.
- Only check whether a shaking block exists; never report RPM/time differences as errors.

## Evaluation criteria
1. **Block matching**: whether each GT block exists in Pred in order (by type).
2. **Parameter accuracy**: whether the parameters of matched block pairs agree.
   - Wait: compare in seconds.
   - Volume, WashCycles, RPM, ShakeTime: must match exactly.
   - **Reservoir number**: if the block type/order match, ignore all reservoir number differences (chained shifting due to Wash Buffer relocation is possible).
3. **Order**: whether the GT order and Pred order agree.
4. **Initial setup block**: only check whether it exists; do not compare internal parameters.
5. **Omission (FN)**: blocks present in GT but absent in Pred.
6. **Over-generation (FP)**: blocks present only in Pred and absent in GT.

## Output format (you must use exactly this JSON format)
```json
{
  "gt_block_count": <number of GT blocks (excluding initial setup)>,
  "pred_block_count": <number of Pred blocks (excluding initial setup)>,
  "matched_count": <number of matched blocks (TP)>,
  "fn_count": <number of blocks present only in GT>,
  "fp_count": <number of blocks present only in Pred>,
  "param_total": <total number of comparable parameters across matched blocks>,
  "param_correct": <number of parameters that match exactly>,
  "order_correct": <number of blocks in correct order>,
  "order_total": <total number of matched blocks>,
  "errors": [
    {
      "type": "missing" | "extra" | "wrong_param" | "wrong_order",
      "gt_block": "<GT block description or null>",
      "pred_block": "<Pred block description or null>",
      "param": "<incorrect parameter name (for wrong_param)>",
      "gt_val": "<GT value>",
      "pred_val": "<Pred value>",
      "reason": "<brief reason>"
    }
  ]
}
```

If there are no errors, output an empty array [].\
"""


_USER_TEMPLATE = """\
Compare the sequence command block lists of GT and Pred below.

=== GT blocks ===
{gt_text}

=== Pred blocks ===
{pred_text}

Respond only in the JSON format specified above.\
"""


def evaluate_sequence_with_llm(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model: str = "claude-sonnet-4-6",
) -> Tuple[CompletenessMetrics, AccuracyMetrics, OrderMetrics, List[dict]]:
    """
    Compare sequence command blocks using the LLM.

    Returns:
        CompletenessMetrics: TP/FN/FP
        AccuracyMetrics: param correct/total
        OrderMetrics: order correct/total
        details: [{errors: [...]}]
    """
    # Parse blocks (rule-based)
    gt_blocks = parse_blocks(gt_df)
    pred_blocks = parse_blocks(pred_df)

    # Exclude initial setup
    gt_main = [b for b in gt_blocks if b.type != "초기설정"]
    pred_main = [b for b in pred_blocks if b.type != "초기설정"]

    gt_text = _blocks_to_text(gt_main)
    pred_text = _blocks_to_text(pred_main)

    # If both are empty
    if not gt_text and not pred_text:
        return CompletenessMetrics(), AccuracyMetrics(), OrderMetrics(), []

    if not gt_text:
        return CompletenessMetrics(), AccuracyMetrics(), OrderMetrics(), []

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
        _raw_response = result_text[:500]  # preserve the raw text for debugging

        # Extract the JSON block (handling the case where it is wrapped in ```json ... ```)
        import re as _re
        _json_block = _re.search(r'```json\s*(.*?)```', result_text, _re.S)
        if not _json_block:
            _json_block = _re.search(r'```\s*(.*?)```', result_text, _re.S)
        if not _json_block:
            # Truncated without a closing ```: extract everything after ```json
            _json_block = _re.search(r'```json\s*(.*)', result_text, _re.S)
        if not _json_block:
            _json_block = _re.search(r'```\s*(.*)', result_text, _re.S)
        if _json_block:
            result_text = _json_block.group(1).strip()

        # Attempt to recover truncated JSON
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # If the errors array is truncated: parse only up to before errors
            _trunc = _re.search(r'(\{.*?"order_total"\s*:\s*\d+)', result_text, _re.S)
            if _trunc:
                result_text = _trunc.group(1) + ', "errors": []}'
                result = json.loads(result_text)
            else:
                raise

    except json.JSONDecodeError as e:
        print(f"    [LLM Sequence] JSON parsing error: {e}", flush=True)
        print(f"    [LLM Sequence] Raw Claude response (first 300 chars):\n      {_raw_response[:300]}", flush=True)
        raise
    except Exception as e:
        print(f"    [LLM Sequence] API error: {type(e).__name__}: {e}", flush=True)
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

    order = OrderMetrics(
        correct_order=result.get("order_correct", 0),
        total_blocks=result.get("order_total", 0),
    )

    errors = result.get("errors", [])

    # Build details
    details = []
    if errors:
        detail_entry = {"section": "Sequence LLM evaluation", "llm_errors": errors}

        fn_list = []
        fp_list = []
        wrong_list = []
        order_list = []
        for err in errors:
            err_type = err.get("type", "")
            reason = err.get("reason", "")
            gt_block = err.get("gt_block", "")
            pred_block = err.get("pred_block", "")

            if err_type == "missing":
                fn_list.append(f"{gt_block} — {reason}")
            elif err_type == "extra":
                fp_list.append(f"{pred_block} — {reason}")
            elif err_type == "wrong_param":
                param = err.get("param", "")
                gt_val = err.get("gt_val", "")
                pred_val = err.get("pred_val", "")
                wrong_list.append(f"{gt_block} → {param}: GT={gt_val} Pred={pred_val}")
            elif err_type == "wrong_order":
                order_list.append(f"{pred_block} — {reason}")

        if fn_list:
            detail_entry["fn_blocks"] = fn_list
        if fp_list:
            detail_entry["fp_blocks"] = fp_list
        if wrong_list:
            detail_entry["wrong_params"] = wrong_list
        if order_list:
            detail_entry["order_errors"] = order_list

        details.append(detail_entry)

    return comp, param_acc, order, details
