"""
structured_evaluator.py — Structured Protocol comparison logic.
Stage 1 (Completeness) + Stage 2 (Parameter Accuracy) + Reservoir Mapping + NATURAL section comparison.
"""

import difflib
from typing import Dict, List, Tuple
from .line_parser import (
    parse_section5, parse_reservoir_mapping,
    extract_instrument_text, extract_natural_text,
    parse_natural_sections, extract_numeric_params,
    extract_user_action_params,
    Action, ReservoirEntry,
)
from .matcher import lcs_match, compare_params
from .metrics import CompletenessMetrics, AccuracyMetrics


def evaluate_structured(gt_text: str, pred_text: str) -> Dict:
    """
    Compare a pair of Structured Protocols.

    Returns:
        {
            "completeness": CompletenessMetrics,
            "parameter_accuracy": AccuracyMetrics,
            "reservoir_accuracy": AccuracyMetrics,
            "completeness_detail": {...},
            "parameter_detail": [...],
            "reservoir_detail": [...],
        }
    """
    # Parse [5] section
    gt_inst = extract_instrument_text(gt_text)
    pred_inst = extract_instrument_text(pred_text)

    gt_actions = parse_section5(gt_inst)
    pred_actions = parse_section5(pred_inst)

    # ── Stage 1: Completeness ──
    matched_pairs, unmatched_gt, unmatched_pred = lcs_match(gt_actions, pred_actions)

    comp = CompletenessMetrics(
        TP=len(matched_pairs),
        FN=len(unmatched_gt),
        FP=len(unmatched_pred),
    )

    # Detail info
    tp_details = []
    for gi, pi in matched_pairs:
        tp_details.append({
            "gt_idx": gi,
            "pred_idx": pi,
            "gt_desc": gt_actions[gi].short_desc(),
            "pred_desc": pred_actions[pi].short_desc(),
        })

    fn_details = [{"gt_idx": i, "gt_desc": gt_actions[i].short_desc(), "gt_line": gt_actions[i].raw_line}
                  for i in unmatched_gt]
    fp_details = [{"pred_idx": j, "pred_desc": pred_actions[j].short_desc(), "pred_line": pred_actions[j].raw_line}
                  for j in unmatched_pred]

    comp_detail = {
        "gt_action_count": len(gt_actions),
        "pred_action_count": len(pred_actions),
        "tp": tp_details,
        "fn": fn_details,
        "fp": fp_details,
    }

    # ══════════════════════════════════════════════════════════════
    # Stage 2: Parameter Accuracy — based on the full GT
    # ══════════════════════════════════════════════════════════════
    # Denominator = parameters of all GT actions (both matched and missing)
    # Numerator = exactly correct parameters
    param_acc_matched = AccuracyMetrics()  # (A) matched-only (for reference)
    param_acc = AccuracyMetrics()          # (B) full-GT basis (primary metric)
    param_details = []

    # The total parameter count of all GT actions is the fixed denominator
    # (independent of Pred)
    _gt_total_params = 0
    for a in gt_actions:
        _gt_total_params += max(len(a.param_dict()), 1)
    param_acc.total = _gt_total_params  # fix the denominator to the GT basis

    # Matched actions: compare parameters
    for gi, pi in matched_pairs:
        correct, total, details = compare_params(gt_actions[gi], pred_actions[pi])
        param_acc_matched.correct += correct
        param_acc_matched.total += total
        param_acc.correct += correct
        # total is already fixed to GT, so it is not added here
        for d in details:
            d["gt_action"] = gt_actions[gi].short_desc()
            d["pred_action"] = pred_actions[pi].short_desc()
            d["gt_raw_line"] = gt_actions[gi].raw_line
            d["pred_raw_line"] = pred_actions[pi].raw_line
        param_details.extend(details)

    # Missing GT actions (FN): correct is 0 (all missing); total is already
    # included in the fixed GT basis
    for gi in unmatched_gt:
        fn_params = gt_actions[gi].param_dict()
        for key, val in fn_params.items():
            param_details.append({
                "param": key, "gt_val": val, "pred_val": "(MISSING)",
                "match": False,
                "gt_action": gt_actions[gi].short_desc(),
                "pred_action": "(not generated)",
                "gt_raw_line": gt_actions[gi].raw_line,
                "pred_raw_line": "-",
            })

    # Over-generated Pred actions (FP): subtract from correct by the number of
    # parameters (penalty)
    fp_penalty = 0
    for pi in unmatched_pred:
        fp_params = pred_actions[pi].param_dict()
        fp_penalty += max(len(fp_params), 1)
    param_acc.correct = max(param_acc.correct - fp_penalty, 0)

    # ── Common: LLM retry settings ──
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _LLM_MAX_RETRIES = 3

    gt_natural = extract_natural_text(gt_text)
    pred_natural = extract_natural_text(pred_text)

    # ── Retry wrapper ──
    from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TE
    _LLM_TIMEOUT = 90  # seconds

    def _run_with_retry(name, fn, fallback_fn=None):
        for _attempt in range(1, _LLM_MAX_RETRIES + 1):
            try:
                print(f"    [LLM {name}] starting call (attempt {_attempt}/{_LLM_MAX_RETRIES}, timeout {_LLM_TIMEOUT}s)...", end="", flush=True)
                _t0 = _time.time()
                with _TPE(max_workers=1) as _exec:
                    _fut = _exec.submit(fn)
                    result = _fut.result(timeout=_LLM_TIMEOUT)
                print(f" done ({_time.time()-_t0:.1f}s)", flush=True)
                return result, "llm"
            except _TE:
                print(f" timed out (exceeded {_LLM_TIMEOUT}s)", flush=True)
                if _attempt < _LLM_MAX_RETRIES:
                    _wait = 3 * _attempt
                    print(f"    [LLM {name}] retrying in {_wait}s...", flush=True)
                    _time.sleep(_wait)
                else:
                    print(f"    [LLM {name}] all attempts failed -> using fallback", flush=True)
                    if fallback_fn:
                        return fallback_fn(), "rule_based"
            except Exception as e:
                print(f" failed ({_time.time()-_t0:.1f}s): {e}", flush=True)
                if _attempt < _LLM_MAX_RETRIES:
                    _wait = 3 * _attempt
                    print(f"    [LLM {name}] retrying in {_wait}s...", flush=True)
                    _time.sleep(_wait)
                else:
                    print(f"    [LLM {name}] all attempts failed -> using fallback", flush=True)
                    if fallback_fn:
                        return fallback_fn(), "rule_based"
                    return None, "rule_only"

    # ── Run 3 LLM evaluations in parallel ──
    from .reservoir_evaluator_llm import evaluate_reservoir_with_llm
    from .natural_evaluator_llm import evaluate_natural_with_llm
    from .instrument_evaluator_llm import evaluate_instrument_with_llm

    res_result = [None, "llm"]
    nat_result = [None, "llm"]
    inst_result = [None, "rule_only"]

    def _eval_reservoir():
        return _run_with_retry(
            "Reservoir",
            lambda: evaluate_reservoir_with_llm(gt_natural, pred_natural),
            lambda: _compare_reservoir_rule_based(gt_natural, pred_natural),
        )

    def _eval_natural():
        return _run_with_retry(
            "Natural",
            lambda: evaluate_natural_with_llm(gt_natural, pred_natural),
            lambda: _compare_natural_sections(gt_natural, pred_natural),
        )

    def _eval_instrument():
        return _run_with_retry(
            "Instrument",
            lambda: evaluate_instrument_with_llm(gt_inst, pred_inst),
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_eval_reservoir): "reservoir",
            executor.submit(_eval_natural): "natural",
            executor.submit(_eval_instrument): "instrument",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result, method = future.result()
                if key == "reservoir":
                    res_result = [result, method]
                elif key == "natural":
                    nat_result = [result, method]
                elif key == "instrument":
                    inst_result = [result, method]
            except Exception as e:
                print(f"    [LLM {key}] unexpected error: {e}")

    # Unpack Reservoir result
    reservoir_eval_method = res_result[1]
    if res_result[0] is not None:
        res_acc, res_details = res_result[0]
    else:
        res_acc, res_details = AccuracyMetrics(), []

    # ── Code-level post-filter for reservoir false positives ──
    res_acc, res_details = _filter_reservoir_llm_false_positives(
        res_acc, res_details, gt_natural, pred_natural
    )

    # Unpack Natural result
    natural_eval_method = nat_result[1]
    if nat_result[0] is not None:
        natural_comp, natural_acc, natural_details = nat_result[0]
    else:
        natural_comp, natural_acc, natural_details = None, None, None

    # Unpack Instrument result
    instrument_eval_method = inst_result[1]
    if inst_result[0] is not None:
        llm_inst_comp, llm_inst_param, llm_inst_details = inst_result[0]
    else:
        llm_inst_comp, llm_inst_param, llm_inst_details = None, None, []

    return {
        # Penalized version (default — includes missing-action penalty)
        "completeness": comp,
        "parameter_accuracy": param_acc,
        # Matched-only version (measured only on matched actions)
        "parameter_accuracy_matched": param_acc_matched,
        # Other metrics
        "reservoir_accuracy": res_acc,
        "natural_completeness": natural_comp,
        "natural_accuracy": natural_acc,
        "completeness_detail": comp_detail,
        "parameter_detail": param_details,
        "reservoir_detail": res_details,
        "natural_detail": natural_details,
        # LLM-based [5] evaluation (alongside rule-based)
        "llm_instrument_completeness": llm_inst_comp,
        "llm_instrument_param_accuracy": llm_inst_param,
        "llm_instrument_detail": llm_inst_details,
        # Evaluation method tracking
        "natural_eval_method": natural_eval_method,
        "instrument_eval_method": instrument_eval_method,
        "reservoir_eval_method": reservoir_eval_method,
    }


# ── Reagent name similarity comparison ──

_REAGENT_SYNONYMS = {
    "wash buffer": ["wash buffer", "wash", "wb"],
    "detection antibody": ["detection antibody", "detection ab", "det antibody", "det ab"],
    "streptavidin hrp": ["streptavidin hrp", "sa hrp", "strep hrp", "streptavidin hrp conjugate"],
    "tmb substrate": ["tmb substrate", "tmb substrate solution", "tmb", "substrate solution", "substrate"],
    "stop solution": ["stop solution", "hcl stop solution", "1m hcl stop solution", "hcl", "1m hcl"],
    "capture antibody": ["capture antibody", "capture ab", "cap antibody", "cap ab"],
    "conjugate": ["conjugate", "enzyme conjugate", "hrp conjugate"],
    "blocking buffer": ["blocking buffer", "block buffer", "blocking"],
    "assay diluent": ["assay diluent", "diluent"],
    "color reagent": ["color reagent", "color reagent a", "color reagent b"],
}


def _reservoir_match(gt_norm: str, pred_norm: str) -> bool:
    """Check if normalized reagent names are semantically equivalent"""
    if gt_norm == pred_norm:
        return True

    # One contains the other
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True

    # Synonym matching
    for canonical, synonyms in _REAGENT_SYNONYMS.items():
        gt_match = any(syn in gt_norm for syn in synonyms)
        pred_match = any(syn in pred_norm for syn in synonyms)
        if gt_match and pred_match:
            return True

    return False


def _filter_reservoir_llm_false_positives(
    acc: AccuracyMetrics,
    details: List[dict],
    gt_natural: str,
    pred_natural: str,
) -> Tuple[AccuracyMetrics, List[dict]]:
    """
    Code-level post-filter for LLM reservoir evaluation results.

    Detects and corrects two types of false positives:
    1. Wash Buffer position shift cascade — when Wash Buffer moves position,
       all other reagents shift reservoir numbers. This is normal, not an error.
    2. User-defined Calibrant bidirectional — either GT or Pred being
       "User-defined" for a Calibrant is acceptable.
    """
    # Only filter error entries (match=False)
    errors = [d for d in details if not d.get("match", False)]
    if not errors:
        return acc, details

    # Parse full reservoir mappings for shift detection
    gt_reservoirs = parse_reservoir_mapping(gt_natural)
    pred_reservoirs = parse_reservoir_mapping(pred_natural)

    gt_map = {r.number: r for r in gt_reservoirs}
    pred_map = {r.number: r for r in pred_reservoirs}

    filtered_details = []
    corrected_count = 0

    for d in details:
        if d.get("match", False):
            filtered_details.append(d)
            continue

        gt_reagent_raw = d.get("gt_reagent", "") or ""
        pred_reagent_raw = d.get("pred_reagent", "") or ""
        res_num = d.get("reservoir", 0)

        # Only correct entries where BOTH sides are present (counted in total).
        # missing_in_pred / extra_in_pred are NOT in total, so correcting them
        # would make correct > total.
        both_present = (
            gt_reagent_raw and gt_reagent_raw != "(missing)"
            and pred_reagent_raw and pred_reagent_raw != "(missing)"
        )

        # ── Filter 1: User-defined Calibrant (bidirectional) ──
        # Only when BOTH sides are Calibrant/Standard type.
        # If one side is a completely different type (Sample, Wash Buffer, etc.),
        # it's a genuine error.
        if both_present:
            gt_lower = gt_reagent_raw.lower()
            pred_lower = pred_reagent_raw.lower()
            gt_is_cal = ("calibrant" in gt_lower or "standard" in gt_lower)
            pred_is_cal = ("calibrant" in pred_lower or "standard" in pred_lower)
            gt_is_ud = ("user-defined" in gt_lower or "사용자 지정" in gt_lower)
            pred_is_ud = ("user-defined" in pred_lower or "사용자 지정" in pred_lower)
            is_user_defined = (gt_is_ud or pred_is_ud) and gt_is_cal and pred_is_cal

            if is_user_defined:
                d_copy = dict(d)
                d_copy["match"] = True
                d_copy["reason"] = "User-defined Calibrant (code-level auto-corrected)"
                filtered_details.append(d_copy)
                corrected_count += 1
                continue

        # ── Filter 2: Wash Buffer position shift cascade ──
        if both_present:
            gt_entry = gt_map.get(res_num)
            pred_entry = pred_map.get(res_num)

            if gt_entry and pred_entry:
                gt_norm = gt_entry.reagent_normalized
                pred_norm = pred_entry.reagent_normalized

                # Check: does the Pred reagent exist somewhere in GT at a different position?
                pred_found_in_gt = any(
                    _reservoir_match(r.reagent_normalized, pred_norm)
                    for r in gt_reservoirs if r.number != res_num
                )
                # Check: does the GT reagent exist somewhere in Pred at a different position?
                gt_found_in_pred = any(
                    _reservoir_match(r.reagent_normalized, gt_norm)
                    for r in pred_reservoirs if r.number != res_num
                )

                if pred_found_in_gt and gt_found_in_pred:
                    d_copy = dict(d)
                    d_copy["match"] = True
                    d_copy["reason"] = "Position shift (code-level auto-corrected)"
                    filtered_details.append(d_copy)
                    corrected_count += 1
                    continue

        # Not a false positive — keep as error
        filtered_details.append(d)

    # Update accuracy counts
    if corrected_count > 0:
        new_acc = AccuracyMetrics()
        new_acc.total = acc.total
        new_acc.correct = min(acc.correct + corrected_count, acc.total)
        print(f"    [Reservoir Filter] {corrected_count} false positive(s) corrected: "
              f"{acc.correct}/{acc.total} -> {new_acc.correct}/{new_acc.total}")
        return new_acc, filtered_details

    return acc, filtered_details


def _compare_reservoir_rule_based(gt_natural: str, pred_natural: str) -> Tuple[AccuracyMetrics, List[dict]]:
    """Rule-based Reservoir mapping comparison (LLM fallback)"""
    gt_reservoirs = parse_reservoir_mapping(gt_natural)
    pred_reservoirs = parse_reservoir_mapping(pred_natural)

    res_acc = AccuracyMetrics()
    res_details = []

    gt_res_map = {r.number: r for r in gt_reservoirs}
    pred_res_map = {r.number: r for r in pred_reservoirs}

    all_res_nums = sorted(set(gt_res_map.keys()) | set(pred_res_map.keys()))
    for num in all_res_nums:
        gt_entry = gt_res_map.get(num)
        pred_entry = pred_res_map.get(num)

        if gt_entry and pred_entry:
            res_acc.total += 1
            # User-defined Calibrant: if either side is user-defined, treat as match
            gt_raw_lower = gt_entry.reagent_raw.lower()
            pred_raw_lower = pred_entry.reagent_raw.lower()
            is_user_defined = (
                ("user-defined" in gt_raw_lower or "사용자 지정" in gt_raw_lower or
                 "user-defined" in pred_raw_lower or "사용자 지정" in pred_raw_lower)
                and ("calibrant" in gt_raw_lower or "standard" in gt_raw_lower
                     or "calibrant" in pred_raw_lower or "standard" in pred_raw_lower)
            )
            match = is_user_defined or _reservoir_match(gt_entry.reagent_normalized, pred_entry.reagent_normalized)
            if match:
                res_acc.correct += 1
            res_details.append({
                "reservoir": num,
                "gt_reagent": gt_entry.reagent_raw,
                "pred_reagent": pred_entry.reagent_raw,
                "match": match,
            })
        elif gt_entry:
            res_acc.total += 1
            res_details.append({
                "reservoir": num,
                "gt_reagent": gt_entry.reagent_raw,
                "pred_reagent": "(missing)",
                "match": False,
            })
        elif pred_entry:
            res_details.append({
                "reservoir": num,
                "gt_reagent": "(missing)",
                "pred_reagent": pred_entry.reagent_raw,
                "match": False,
            })

    return res_acc, res_details


def _normalize_time_params(params: list) -> list:
    """
    Normalize time parameters: unify to minutes.
    - ("1", "시간") -> ("60", "분")
    - ("2", "시간") -> ("120", "분")
    - ("0.5", "시간") -> ("30", "분")
    This ensures "60 분" and "1 시간" match as equivalent.
    (Note: "시간" = hours, "분" = minutes; these unit strings are data
    values and are preserved verbatim.)
    """
    result = []
    for val, unit in params:
        if unit == "시간":
            try:
                minutes = float(val) * 60
                # If integer, use integer string; otherwise keep as-is
                if minutes == int(minutes):
                    result.append((str(int(minutes)), "분"))
                else:
                    result.append((str(minutes), "분"))
            except ValueError:
                result.append((val, unit))
        else:
            result.append((val, unit))
    return result


def _compare_natural_sections(gt_natural: str, pred_natural: str) -> Tuple[CompletenessMetrics, AccuracyMetrics, List[dict]]:
    """
    Compare NATURAL sections [1]~[3] by parameter **existence (set)**.

    [1]~[3] are free-form text, so GT and Pred may describe the same
    information in different order/structure. Instead of line-by-line diff:

    1) Extract all (value, unit) pairs from the entire [1]~[3] text
    2) Compare GT set vs Pred set (ignoring duplicate counts)
       - Parameters only in GT -> missing (FN)
       - Parameters only in Pred -> over-generated (FP)
       - Common to both -> correct (TP)
    3) Completeness: TP / FP / FN -> F1
       Accuracy: common / GT unique count (GT-based recall)

    Note: If the same parameter (e.g., 300 uL) appears 3 times in GT
    and 1 time in Pred, it is not considered missing as long as it is
    mentioned at least once -> set comparison.
    (Omitting repeated mentions is not an error)

    Cross-section movement (parameter in GT [1] appears in Pred [2])
    is allowed -- compare as a whole pool regardless of section placement.

    Returns:
        CompletenessMetrics: TP/FP/FN -> F1
        AccuracyMetrics: correct/total
        details: [{section, missing_from_pred, extra_in_pred,
                   missing_source_lines, extra_source_lines}]
    """
    gt_secs = parse_natural_sections(gt_natural)
    pred_secs = parse_natural_sections(pred_natural)

    # Combine all of [1]~[3] into a single pool
    gt_param_set = set()
    pred_param_set = set()

    # Parameter -> source line mapping (for traceability)
    gt_param_to_lines: Dict[tuple, List[str]] = {}
    pred_param_to_lines: Dict[tuple, List[str]] = {}

    import re
    _ASPIRATE_PATTERN = re.compile(r'용액\s*제거|aspirat', re.IGNORECASE)

    for sec_num in ("1", "2", "3"):
        for line in gt_secs.get(sec_num, []):
            params = extract_user_action_params(line)
            normed = _normalize_time_params(params)
            # Exclude the uL parameters of aspirate (solution removal) lines from evaluation
            if _ASPIRATE_PATTERN.search(line):
                normed = [p for p in normed if p[1] != 'µL']
            for p in normed:
                gt_param_set.add(p)
                gt_param_to_lines.setdefault(p, []).append(line.strip())
        for line in pred_secs.get(sec_num, []):
            params = extract_user_action_params(line)
            normed = _normalize_time_params(params)
            if _ASPIRATE_PATTERN.search(line):
                normed = [p for p in normed if p[1] != 'µL']
            for p in normed:
                pred_param_set.add(p)
                pred_param_to_lines.setdefault(p, []).append(line.strip())

    # Set comparison: unique types
    common = gt_param_set & pred_param_set
    gt_only = gt_param_set - pred_param_set
    pred_only = pred_param_set - gt_param_set
    gt_total = len(gt_param_set)
    common_count = len(common)

    # Completeness: TP/FP/FN
    comp = CompletenessMetrics()
    comp.TP = common_count
    comp.FP = len(pred_only)
    comp.FN = len(gt_only)

    acc = AccuracyMetrics()
    acc.total = gt_total
    acc.correct = common_count

    # Difference details
    details = []
    missing_from_pred = gt_only
    extra_in_pred = pred_only

    if missing_from_pred or extra_in_pred:
        detail_entry = {"section": "[1]~[3] combined"}
        if missing_from_pred:
            detail_entry["missing_from_pred"] = [
                f"{v} {u}" for (v, u) in sorted(missing_from_pred)
            ]
            # Trace which GT source lines the missing parameters came from
            missing_lines = []
            for p in sorted(missing_from_pred):
                for sl in gt_param_to_lines.get(p, []):
                    missing_lines.append(sl)
            detail_entry["missing_source_lines"] = list(dict.fromkeys(missing_lines))
        if extra_in_pred:
            detail_entry["extra_in_pred"] = [
                f"{v} {u}" for (v, u) in sorted(extra_in_pred)
            ]
            extra_lines = []
            for p in sorted(extra_in_pred):
                for sl in pred_param_to_lines.get(p, []):
                    extra_lines.append(sl)
            detail_entry["extra_source_lines"] = list(dict.fromkeys(extra_lines))
        details.append(detail_entry)

    return comp, acc, details
