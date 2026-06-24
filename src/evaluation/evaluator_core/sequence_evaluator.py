"""
sequence_evaluator.py — Sequence (.xlsx) comparison logic.
Stage 1 (Completeness) + Stage 2 (Parameter Accuracy) + Stage 3 (Execution Order).
"""

from typing import Dict, List
import pandas as pd
from .block_parser import parse_blocks, SeqBlock
from .matcher import lcs_match, compare_params, check_order_match
from .metrics import CompletenessMetrics, AccuracyMetrics, OrderMetrics


def evaluate_sequence(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict:
    """
    Compare a pair of Sequence xlsx files.

    Returns:
        {
            "completeness": CompletenessMetrics,
            "parameter_accuracy": AccuracyMetrics,
            "execution_order": OrderMetrics,
            "completeness_detail": {...},
            "parameter_detail": [...],
            "order_detail": {...},
        }
    """
    # Block segmentation
    gt_blocks = parse_blocks(gt_df)
    pred_blocks = parse_blocks(pred_df)

    # ── Stage 1: Completeness (block-level) ──
    matched_pairs, unmatched_gt, unmatched_pred = lcs_match(gt_blocks, pred_blocks)

    comp = CompletenessMetrics(
        TP=len(matched_pairs),
        FN=len(unmatched_gt),
        FP=len(unmatched_pred),
    )

    tp_details = []
    for gi, pi in matched_pairs:
        tp_details.append({
            "gt_idx": gi,
            "pred_idx": pi,
            "gt_desc": gt_blocks[gi].short_desc(),
            "pred_desc": pred_blocks[pi].short_desc(),
            "gt_rows": f"{gt_blocks[gi].start_idx}-{gt_blocks[gi].end_idx}",
            "pred_rows": f"{pred_blocks[pi].start_idx}-{pred_blocks[pi].end_idx}",
        })

    fn_details = [{"gt_idx": i, "gt_desc": gt_blocks[i].short_desc(),
                    "gt_rows": f"{gt_blocks[i].start_idx}-{gt_blocks[i].end_idx}"}
                  for i in unmatched_gt]
    fp_details = [{"pred_idx": j, "pred_desc": pred_blocks[j].short_desc(),
                    "pred_rows": f"{pred_blocks[j].start_idx}-{pred_blocks[j].end_idx}"}
                  for j in unmatched_pred]

    comp_detail = {
        "gt_block_count": len(gt_blocks),
        "pred_block_count": len(pred_blocks),
        "tp": tp_details,
        "fn": fn_details,
        "fp": fp_details,
    }

    # ══════════════════════════════════════════════════════════════
    # Stage 2: Parameter Accuracy — based on the full GT (no weighting, raw parameters)
    # ══════════════════════════════════════════════════════════════
    # Denominator = actual parameter count of all GT blocks
    # Numerator = number of exactly correct parameters
    # No weighting/coverage. Use only what compare_params() returns.
    #
    # (A) matched-only: only on matched blocks (for reference)
    # (B) gt_total: full-GT basis (primary metric) — FN block parameters are also included in total
    param_acc_matched = AccuracyMetrics()
    param_acc = AccuracyMetrics()
    param_details = []

    # The total parameter count of all GT blocks is the fixed denominator
    # (independent of Pred, never changes)
    _gt_total_params = 0
    for b in gt_blocks:
        gt_rows = b.end_idx - b.start_idx + 1
        _gt_total_params += gt_rows + max(len(b.param_dict()), 1)
    param_acc.total = _gt_total_params  # fix the denominator to the GT basis

    # ── Matched blocks: compare parameters ──
    for gi, pi in matched_pairs:
        correct, total, details = compare_params(gt_blocks[gi], pred_blocks[pi])
        param_acc_matched.correct += correct
        param_acc_matched.total += total
        param_acc.correct += correct
        # total is already fixed to GT, so it is not added here

        for d in details:
            d["gt_block"] = gt_blocks[gi].short_desc()
            d["pred_block"] = pred_blocks[pi].short_desc()
            d["gt_rows"] = f"{gt_blocks[gi].start_idx}-{gt_blocks[gi].end_idx}"
            d["pred_rows"] = f"{pred_blocks[pi].start_idx}-{pred_blocks[pi].end_idx}"
        param_details.extend(details)

    # ── Missing GT blocks (FN): correct is 0 (all missing); total is already included in the fixed GT basis ──
    for gi in unmatched_gt:
        fn_params = gt_blocks[gi].param_dict()

        for key, val in fn_params.items():
            param_details.append({
                "param": key, "gt_val": val, "pred_val": "(MISSING)",
                "match": False,
                "gt_block": gt_blocks[gi].short_desc(),
                "pred_block": "(not generated)",
                "gt_rows": f"{gt_blocks[gi].start_idx}-{gt_blocks[gi].end_idx}",
                "pred_rows": "-",
            })

    # Over-generated Pred blocks (FP): subtract from correct by parameter count + row count (penalty)
    fp_penalty = 0
    for pi in unmatched_pred:
        fp_rows = pred_blocks[pi].end_idx - pred_blocks[pi].start_idx + 1
        fp_params = pred_blocks[pi].param_dict()
        fp_penalty += fp_rows + max(len(fp_params), 1)
    param_acc.correct = max(param_acc.correct - fp_penalty, 0)

    # ══════════════════════════════════════════════════════════════
    # Stage 3: Execution Order — based on the full GT
    # ══════════════════════════════════════════════════════════════
    correct_order, total_matched = check_order_match(gt_blocks, pred_blocks, matched_pairs)
    violations = _check_constraint_violations(pred_df)

    # (A) matched-only order (for reference)
    order_matched = OrderMetrics(
        correct_order=correct_order,
        total_blocks=total_matched,
        constraint_violations=violations,
    )
    # (B) full-GT basis: missing blocks count as order errors
    order = OrderMetrics(
        correct_order=correct_order,
        total_blocks=len(gt_blocks),  # the total GT block count is the denominator
        constraint_violations=violations,
    )

    order_detail = {
        "gt_block_order": [b.short_desc() for b in gt_blocks],
        "pred_block_order": [b.short_desc() for b in pred_blocks],
        "constraint_violations": violations,
    }

    # ── LLM-based sequence evaluation (alongside rule-based, with retry) ──
    import time as _time

    _LLM_MAX_RETRIES = 3
    llm_seq_comp = None
    llm_seq_param = None
    llm_seq_order = None
    llm_seq_details = []
    sequence_eval_method = "rule_only"

    from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TE
    _LLM_TIMEOUT = 90  # seconds

    for _attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            print(f"    [LLM Sequence] starting call (attempt {_attempt}/{_LLM_MAX_RETRIES}, timeout {_LLM_TIMEOUT}s)...", end="", flush=True)
            _t0 = _time.time()
            from .sequence_evaluator_llm import evaluate_sequence_with_llm

            with _TPE(max_workers=1) as _exec:
                _fut = _exec.submit(evaluate_sequence_with_llm, gt_df, pred_df)
                llm_seq_comp, llm_seq_param, llm_seq_order, llm_seq_details = _fut.result(timeout=_LLM_TIMEOUT)

            print(f" done ({_time.time()-_t0:.1f}s)", flush=True)
            sequence_eval_method = "llm"
            break
        except _TE:
            print(f" timed out (exceeded {_LLM_TIMEOUT}s)", flush=True)
            if _attempt < _LLM_MAX_RETRIES:
                _wait = 3 * _attempt
                print(f"    [LLM Sequence] retrying in {_wait}s...", flush=True)
                _time.sleep(_wait)
        except Exception as e:
            print(f" failed ({_time.time()-_t0:.1f}s): {e}", flush=True)
            if _attempt < _LLM_MAX_RETRIES:
                _wait = 3 * _attempt
                print(f"    [LLM Sequence] retrying in {_wait}s...", flush=True)
                _time.sleep(_wait)
            else:
                print(f"    [LLM Sequence] all attempts failed -> using rule-based fallback", flush=True)

    return {
        # Penalized version (default — includes missing-block penalty)
        "completeness": comp,
        "parameter_accuracy": param_acc,
        "execution_order": order,
        # Matched-only version (measured only on matched blocks)
        "parameter_accuracy_matched": param_acc_matched,
        "execution_order_matched": order_matched,
        # Details
        "completeness_detail": comp_detail,
        "parameter_detail": param_details,
        "order_detail": order_detail,
        # LLM-based evaluation (alongside rule-based)
        "llm_completeness": llm_seq_comp,
        "llm_parameter_accuracy": llm_seq_param,
        "llm_execution_order": llm_seq_order,
        "llm_detail": llm_seq_details,
        # Evaluation method tracking
        "sequence_eval_method": sequence_eval_method,
    }


def _check_constraint_violations(df: pd.DataFrame) -> int:
    """
    Check for physical constraint violations:
    1. ASPIRATE/DISPENSE without prior INSTALL (tip not installed)
    2. INSTALL without prior EJECT (double install = tip collision)
    3. Channel mismatch: 4ch tip coordinates used in 1ch mode or vice versa
    4. Invalid command (not in allowed set)
    """
    ALLOWED_COMMANDS = {
        "A#GET", "A#PUT", "A#TEMPPLATE_ON", "A#TEMPPLATE_OFF",
        "A#CAP_OPEN", "A#CAP_CLOSE",
        "A#ADP_SELECT_1_CHANNEL", "A#ADP_SELECT_4_CHANNEL",
        "A#ADP_INSTALL_PIPETTE", "A#ADP_EJECT_PIPETTE",
        "A#ADP_ASPIRATE_FROM_PLATE", "A#ADP_DISPENSE_INTO_PLATE",
        "A#TRANS", "A#SHAKE_START", "A#WAIT_MOTION",
        "A#ANALYZER_OPEN", "A#ANALYZER_CLOSE", "A#ANALYZER_START",
    }

    violations = 0
    tip_installed = False
    current_channel = None  # 4 or 1

    for _, row in df.iterrows():
        cmd = str(row.get("Command", "")).strip()
        params = str(row.get("Input Parameters", "")).strip()

        # 1. Invalid command check
        if cmd and cmd.startswith("A#") and cmd not in ALLOWED_COMMANDS:
            violations += 1
            continue

        if cmd == "A#ADP_SELECT_4_CHANNEL":
            current_channel = 4
        elif cmd == "A#ADP_SELECT_1_CHANNEL":
            current_channel = 1
        elif cmd == "A#ADP_INSTALL_PIPETTE":
            # 2. Double install check (tip already installed without eject)
            if tip_installed:
                violations += 1
            tip_installed = True

            # 3. Channel-tip coordinate validation
            if current_channel is not None and params:
                parts = params.split()
                if len(parts) >= 3:
                    try:
                        tip_row = int(parts[2])  # Y coordinate
                        if current_channel == 4 and tip_row not in (1, 5):
                            violations += 1  # 4ch must start at row 1 or 5
                    except ValueError:
                        pass

        elif cmd == "A#ADP_EJECT_PIPETTE":
            tip_installed = False
        elif cmd in ("A#ADP_ASPIRATE_FROM_PLATE", "A#ADP_DISPENSE_INTO_PLATE"):
            # 1. No tip installed
            if not tip_installed:
                violations += 1

    # 5. Reservoir gap check (number continuity)
    reservoir_nums = set()
    for _, row in df.iterrows():
        cmd = str(row.get("Command", "")).strip()
        params = str(row.get("Input Parameters", "")).strip()
        if cmd == "A#ADP_ASPIRATE_FROM_PLATE" and params:
            parts = params.split()
            if len(parts) >= 2 and parts[0] == "5":  # slot 5 = reservoir
                try:
                    reservoir_nums.add(int(parts[1]))
                except ValueError:
                    pass

    if reservoir_nums:
        max_res = max(reservoir_nums)
        for i in range(1, max_res + 1):
            if i not in reservoir_nums:
                violations += 1  # Reservoir gap detected

    return violations
