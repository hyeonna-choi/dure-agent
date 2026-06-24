"""
evaluator.py — Performance evaluation script.

Features:
  1) All models x before (v0_initial) performance evaluation + 5 report files
  2) All models x all phases (before/attempt_1/attempt_2/attempt_3) validation loop evaluation + 5 report files
  3) Console Table 1/2/3/detail table output
  4) CSV export: model_comparison.csv, validation_loop_metrics.csv,
     validation_loop_metrics_full.csv, validation_loop_metrics_by_phase.csv
  5) Aggregated error log: aggregated_errors.txt

Usage:
  python evaluator.py
"""

import os
import re
import csv
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import pandas as pd

# Project root (the `src` directory) so that `bioforge` can be imported and all
# data/output paths are independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator_core.structured_evaluator import evaluate_structured
from evaluator_core.sequence_evaluator import evaluate_sequence
from evaluator_core.metrics import (
    CompletenessMetrics, AccuracyMetrics, OrderMetrics,
    aggregate_completeness, aggregate_accuracy, aggregate_order,
)
from evaluator_core.report_writer import write_reports


# ══════════════════════════════════════════════════════════
#                     Path Settings

# Ground Truth (GT) folder
GT_STRUCTURED_DIR = PROJECT_ROOT / "data" / "protocol_gt" / "structuredProtocol"
GT_SEQUENCE_DIR   = PROJECT_ROOT / "data" / "protocol_gt" / "sequences"

# Model results folder: if V0_REVALIDATION is set, auto-select output_{validator_model}/results
def _resolve_results_root() -> Path:
    try:
        from bioforge.config.settings import Config
        _cfg = Config()
        _base = PROJECT_ROOT / "output"
        if getattr(_cfg, "V0_REVALIDATION", False):
            _vml = _cfg.get_validator_model_list()
            _v_safe = _vml[0].replace("/", "_").replace(":", "_")
            return _base.parent / f"{_base.name}_{_v_safe}" / "results"
    except Exception:
        pass
    return PROJECT_ROOT / "output" / "results"

RESULTS_ROOT = _resolve_results_root()

# Evaluation results output folder
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

# ════════════════════════════════════════════════════════════
# ── File Matching ──

STRUCTURED_SUFFIX = "_StructuredProtocol.txt"
SEQUENCE_SUFFIX = "_Seq.xlsx"

# v0_initial file suffix within validation/versions/
V0_STRUCTURED_SUFFIX = "_StructuredProtocol_v0_initial.txt"
V0_SEQUENCE_SUFFIX = "_Seq_v0_initial.xlsx"

# Exclude common folders like graph_data, timing_reports; collect only model folders
_SKIP_DIRS = {"graph_data", "timing_reports"}

# Per-phase version labels
VERSION_LABELS = ["before", "attempt_1", "attempt_2", "attempt_3"]

# Per-phase version tags (priority order) — for run_validation_loop_metrics
VERSION_TAGS = {
    "before":    ["v0_initial"],
    "attempt_1": ["v1_PASS_final", "v1_after", "v1_FAIL_final"],
    "attempt_2": ["v2_PASS_final", "v2_after", "v2_FAIL_final"],
    "attempt_3": ["v3_PASS_final", "v3_after", "v3_FAIL_final"],
}


# ══════════════════════════════════════════════════════════
#                    Utility Functions

def _extract_stem(filename: str, suffix: str) -> str:
    """Extract stem from filename: '0120T018_StructuredProtocol.txt' -> '0120T018'"""
    if filename.endswith(suffix):
        return filename[:-len(suffix)]
    return Path(filename).stem


def _discover_files(folder: Path, suffix: str, sub_folder: Optional[str] = None) -> Dict[str, Path]:
    """
    Discover files in folder. Supports both flat and nested structures.

    flat: folder/0120T018_StructuredProtocol.txt
    nested: folder/0120T018/structured/0120T018_StructuredProtocol.txt
    nested(v0): folder/0120T018/validation/versions/0120T018_StructuredProtocol_v0_initial.txt
    """
    found = {}

    # 1) flat structure: files directly under folder
    for f in folder.glob(f"*{suffix}"):
        stem = _extract_stem(f.name, suffix)
        found[stem] = f

    # 2) nested structure: folder/stem/sub_folder/stem_suffix
    if sub_folder:
        for d in folder.iterdir():
            if d.is_dir():
                sub = d / sub_folder
                if sub.exists() and sub.is_dir():
                    for f in sub.glob(f"*{suffix}"):
                        stem = _extract_stem(f.name, suffix)
                        if stem not in found:
                            found[stem] = f

    return found


def match_files(
    gt_dir: Optional[Path],
    pred_dir: Optional[Path],
    gt_suffix: str = "",
    pred_suffix: str = "",
    gt_sub_folder: Optional[str] = None,
    pred_sub_folder: Optional[str] = None,
    # backward compatibility
    suffix: str = "",
    sub_folder: Optional[str] = None,
) -> Tuple[Dict[str, Tuple[Path, Path]], List[str], List[str]]:
    """
    Match files from GT and Pred folders by stem.

    Returns:
        matched: {stem: (gt_path, pred_path)}
        unmatched_gt: [stem, ...]
        unmatched_pred: [stem, ...]
    """
    if not gt_dir or not pred_dir:
        return {}, [], []

    _gt_suffix = gt_suffix or suffix
    _pred_suffix = pred_suffix or suffix
    _gt_sub_folder = gt_sub_folder
    _pred_sub_folder = pred_sub_folder or sub_folder

    gt_files = _discover_files(gt_dir, _gt_suffix, sub_folder=_gt_sub_folder)
    pred_files = _discover_files(pred_dir, _pred_suffix, sub_folder=_pred_sub_folder)

    all_stems = set(gt_files.keys()) | set(pred_files.keys())
    matched = {}
    unmatched_gt = []
    unmatched_pred = []

    for stem in sorted(all_stems):
        if stem in gt_files and stem in pred_files:
            matched[stem] = (gt_files[stem], pred_files[stem])
        elif stem in gt_files:
            unmatched_gt.append(stem)
        else:
            unmatched_pred.append(stem)

    return matched, unmatched_gt, unmatched_pred


def _auto_discover_model_dirs(results_root: Path = RESULTS_ROOT) -> dict:
    """Auto-discover model folders under results_root and return {model_name: Path}"""
    model_dirs = {}
    if not results_root.exists():
        return model_dirs
    for d in sorted(results_root.iterdir()):
        if d.is_dir() and d.name not in _SKIP_DIRS:
            model_dirs[d.name] = d
    return model_dirs


def _find_version_file_by_num(versions_dir: Path, stem: str, version_num: int,
                               file_type: str) -> Optional[Path]:
    """
    Find the file for a given version.
    version_num: 0=v0_initial, 1=after attempt_1, 2=after attempt_2, 3=after attempt_3

    Priority:
      v0: v0_initial
      v1~v3: PASS_final > after > FAIL_final
    """
    if file_type == "structured":
        base_suffix = "_StructuredProtocol"
        ext = ".txt"
    else:
        base_suffix = "_Seq"
        ext = ".xlsx"

    if version_num == 0:
        candidates = [f"{stem}{base_suffix}_v0_initial{ext}"]
    else:
        v = version_num
        candidates = [
            f"{stem}{base_suffix}_v{v}_PASS_final{ext}",
            f"{stem}{base_suffix}_v{v}_after{ext}",
            f"{stem}{base_suffix}_v{v}_FAIL_final{ext}",
        ]

    for cand in candidates:
        p = versions_dir / cand
        if p.exists():
            return p
    return None


def _find_best_version(versions_dir: Path, stem: str, version_num: int,
                       file_type: str) -> Optional[Path]:
    """
    Find the most recent file at or below the given version.
    e.g.: if version_num=2 but v2 doesn't exist, use v1_PASS_final (already passed)
    """
    f = _find_version_file_by_num(versions_dir, stem, version_num, file_type)
    if f:
        return f
    # If not found, search previous versions
    for v in range(version_num - 1, -1, -1):
        f = _find_version_file_by_num(versions_dir, stem, v, file_type)
        if f:
            return f
    return None


def _find_version_file_by_tags(version_dir: Path, stem: str, file_type: str,
                                tags: List[str]) -> Optional[Path]:
    """
    Find file based on tag list.
    Priority: PASS_final > after > FAIL_final
    """
    if file_type == "structured":
        suffix_base = "_StructuredProtocol"
        ext = ".txt"
    else:
        suffix_base = "_Seq"
        ext = ".xlsx"

    for tag in tags:
        filename = f"{stem}{suffix_base}_{tag}{ext}"
        path = version_dir / filename
        if path.exists():
            return path
    return None


def fmt(val, digits=4):
    """Number formatting utility"""
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _fmt_csv(v):
    """CSV value formatting"""
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 4)
    return v


# ══════════════════════════════════════════════════════════
#             Single model v0_initial evaluation (Table 1 & 2)

def evaluate_model(model_name: str, model_dir: Path, output_dir: Path) -> dict:
    """
    Single model evaluation (v0_initial basis) -> aggregate results + 5 report files.

    Returns:
        dict with all metrics (for Table 1&2 + Table 3 v0_metrics)
    """
    if not model_dir.exists():
        print(f"  [SKIP] {model_name}: results folder not found")
        return {}

    # Direct mode: look up files directly in the sequences/ folder (no validation/versions)
    is_direct = model_name.endswith("-direct")

    if is_direct:
        # Direct mode: no structured output, sequence only from the sequences/ folder
        struct_matched, struct_unmatched_gt, struct_unmatched_pred = {}, [], []
        seq_matched, seq_unmatched_gt, seq_unmatched_pred = match_files(
            GT_SEQUENCE_DIR, model_dir,
            gt_suffix=SEQUENCE_SUFFIX,
            pred_suffix=SEQUENCE_SUFFIX,
            pred_sub_folder="sequences",
        )
    else:
        # Standard mode: locate v0_initial files
        # Old layout: validation/versions/   New layout: validation/{validator_model}/versions/
        def _resolve_v0_sub_folder(model_dir: Path, sample_stem: str) -> str:
            """Inspect one sample file to return the actual versions folder path"""
            old_path = model_dir / sample_stem / "validation" / "versions"
            if old_path.exists():
                return "validation/versions"
            val_dir = model_dir / sample_stem / "validation"
            if val_dir.exists():
                for sub in val_dir.iterdir():
                    if sub.is_dir():
                        new_path = sub / "versions"
                        if new_path.exists():
                            return f"validation/{sub.name}/versions"
            return "validation/versions"  # fallback

        # Find a sample stem (the first protocol folder)
        _sample_stem = next(
            (d.name for d in model_dir.iterdir() if d.is_dir()), ""
        )
        _v0_sub = _resolve_v0_sub_folder(model_dir, _sample_stem) if _sample_stem else "validation/versions"

        struct_matched, struct_unmatched_gt, struct_unmatched_pred = match_files(
            GT_STRUCTURED_DIR, model_dir,
            gt_suffix=STRUCTURED_SUFFIX,
            pred_suffix=V0_STRUCTURED_SUFFIX,
            pred_sub_folder=_v0_sub,
        )
        seq_matched, seq_unmatched_gt, seq_unmatched_pred = match_files(
            GT_SEQUENCE_DIR, model_dir,
            gt_suffix=SEQUENCE_SUFFIX,
            pred_suffix=V0_SEQUENCE_SUFFIX,
            pred_sub_folder=_v0_sub,
        )

    # Aggregation lists
    s_comp, s_param, s_res, s_nat, s_nat_comp = [], [], [], [], []
    q_comp, q_param, q_order = [], [], []
    q_param_matched, q_order_matched = [], []  # matched-only version
    # GT-fixed rule-based only — fixes the Overall_Accuracy denominator
    s_param_gt, q_param_gt = [], []
    per_file_results = {}

    all_stems = sorted(set(list(struct_matched.keys()) + list(seq_matched.keys())))
    _total = len(all_stems)
    import time as _eval_time

    _eval_start = _eval_time.time()

    # Real-time log file + intermediate results JSON
    _log_dir = Path(output_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)
    _error_log_path = _log_dir / f"{model_name}_eval_log.txt"
    _partial_json_path = _log_dir / f"{model_name}_partial_results.json"

    with open(_error_log_path, "w", encoding="utf-8") as f:
        f.write(f"=== {model_name} Evaluation Log ===\n\n")

    print(f"  Evaluating {_total} files (sequential processing):")
    print(f"  Log: {_error_log_path}")

    for i, stem in enumerate(all_stems):
        _file_start = _eval_time.time()
        print(f"    [{i+1}/{_total}] {stem} ...", end="", flush=True)

        file_result = {}

        # Structured
        if stem in struct_matched:
            gt_path, pred_path = struct_matched[stem]
            try:
                gt_text = gt_path.read_text(encoding="utf-8", errors="ignore")
                pred_text = pred_path.read_text(encoding="utf-8", errors="ignore")
                sr = evaluate_structured(gt_text, pred_text)
                file_result["structured"] = sr
            except Exception as e:
                file_result["error"] = f"Structured: {e}"

        # Sequence
        if stem in seq_matched:
            gt_path, pred_path = seq_matched[stem]
            try:
                gt_df = pd.read_excel(gt_path)
                pred_df = pd.read_excel(pred_path)
                qr = evaluate_sequence(gt_df, pred_df)
                file_result["sequence"] = qr
            except Exception as e:
                err_msg = f"Sequence: {e}"
                if "error" in file_result:
                    file_result["error"] += f" | {err_msg}"
                else:
                    file_result["error"] = err_msg

        # Progress + estimated time remaining
        _file_elapsed = _eval_time.time() - _file_start
        _total_elapsed = _eval_time.time() - _eval_start
        _avg = _total_elapsed / (i + 1)
        _remaining = _avg * (_total - i - 1)
        _rem_min = int(_remaining // 60)
        _rem_sec = int(_remaining % 60)
        _status = "OK" if "error" not in file_result else "ERR"
        print(f" [{_status}] ({_file_elapsed:.1f}s)  remaining ~{_rem_min}m {_rem_sec}s", flush=True)

        # On each file completion, write to the log immediately
        with open(_error_log_path, "a", encoding="utf-8") as _lf:
            _lf.write(f"[{i+1}/{_total}] {stem} [{_status}] ({_file_elapsed:.1f}s)\n")
            if "error" in file_result:
                _lf.write(f"  ERROR: {file_result['error']}\n")
            if "sequence" in file_result:
                _qr = file_result["sequence"]
                _llm_d = _qr.get("llm_detail", [])
                if _llm_d:
                    _lf.write(f"  Seq Errors:\n")
                    for _d in _llm_d:
                        if isinstance(_d, dict):
                            for _err in _d.get("errors", []):
                                _lf.write(f"    - {_err}\n")
                _sc = _qr.get("llm_completeness") or _qr.get("completeness")
                _sp = _qr.get("llm_parameter_accuracy") or _qr.get("parameter_accuracy")
                _so = _qr.get("llm_execution_order") or _qr.get("execution_order")
                if _sc:
                    _lf.write(f"  Completeness: TP={_sc.TP} FP={_sc.FP} FN={_sc.FN} F1={_sc.f1:.4f}\n")
                if _sp:
                    _lf.write(f"  Param Acc: {_sp.correct}/{_sp.total} = {_sp.accuracy:.4f}\n")
                if _so:
                    _lf.write(f"  Order Acc: {_so.correct_order}/{_so.total_blocks} = {_so.order_accuracy:.4f}\n")
            _lf.write("\n")

        # Aggregate results
        if file_result:
            per_file_results[stem] = file_result

            if "structured" in file_result:
                sr = file_result["structured"]
                s_comp.append(sr.get("llm_instrument_completeness") or sr["completeness"])
                s_param.append(sr.get("llm_instrument_param_accuracy") or sr["parameter_accuracy"])
                s_param_gt.append(sr["parameter_accuracy"])  # GT-fixed rule-based only
                s_res.append(sr["reservoir_accuracy"])
                s_nat.append(sr["natural_accuracy"])
                s_nat_comp.append(sr["natural_completeness"])

            if "sequence" in file_result:
                qr = file_result["sequence"]
                q_comp.append(qr.get("llm_completeness") or qr["completeness"])
                q_param.append(qr.get("llm_parameter_accuracy") or qr["parameter_accuracy"])
                q_param_gt.append(qr["parameter_accuracy"])  # GT-fixed rule-based only
                q_order.append(qr.get("llm_execution_order") or qr["execution_order"])
                # matched-only version (collect if present, otherwise same as penalized)
                q_param_matched.append(qr.get("parameter_accuracy_matched") or qr["parameter_accuracy"])
                q_order_matched.append(qr.get("execution_order_matched") or qr["execution_order"])

        # Save intermediate results JSON (allows recovery if interrupted)
        import json as _json
        _partial = {
            "model": model_name,
            "completed": i + 1,
            "total": _total,
            "elapsed_sec": round(_total_elapsed, 1),
        }
        if q_comp:
            _f1s = [c.f1 for c in q_comp]
            _partial["avg_seq_f1"] = round(sum(_f1s) / len(_f1s), 4)
        if q_param:
            _accs = [p.accuracy for p in q_param]
            _partial["avg_seq_param_acc"] = round(sum(_accs) / len(_accs), 4)
        with open(_partial_json_path, "w", encoding="utf-8") as _jf:
            _json.dump(_partial, _jf, indent=2, ensure_ascii=False)

    print()  # newline

    result = {
        "n_structured": len(struct_matched),
        "n_sequence": len(seq_matched),
    }

    # ── Structured aggregation ──
    structured_agg = None
    if s_comp:
        sa = aggregate_completeness(s_comp)
        result["inst_comp_f1"] = sa["f1"]
        result["inst_comp_p"] = sa["precision"]
        result["inst_comp_r"] = sa["recall"]
        result["inst_comp_TP"] = sa["TP"]
        result["inst_comp_FP"] = sa["FP"]
        result["inst_comp_FN"] = sa["FN"]

        sp = aggregate_accuracy(s_param)
        result["inst_param_acc"] = sp["accuracy"]
        # Fix the Overall_Accuracy denominator to the GT-fixed rule-based value
        sp_gt = aggregate_accuracy(s_param_gt) if s_param_gt else sp
        result["inst_param_correct"] = sp_gt["correct"]
        result["inst_param_total"] = sp_gt["total"]

        sr_agg = aggregate_accuracy(s_res)
        result["res_acc"] = sr_agg["accuracy"]
        result["res_correct"] = sr_agg["correct"]
        result["res_total"] = sr_agg["total"]

        sn = aggregate_accuracy(s_nat)
        result["nat_param_acc"] = sn["accuracy"]
        result["nat_param_correct"] = sn["correct"]
        result["nat_param_total"] = sn["total"]

        snc = aggregate_completeness(s_nat_comp)
        result["nat_comp_f1"] = snc["f1"]
        result["nat_comp_p"] = snc["precision"]
        result["nat_comp_r"] = snc["recall"]
        result["nat_comp_TP"] = snc["TP"]
        result["nat_comp_FP"] = snc["FP"]
        result["nat_comp_FN"] = snc["FN"]

        structured_agg = {
            "stage1_completeness": sa,
            "stage2_parameter_accuracy": sp,
            "reservoir_mapping": sr_agg,
            "natural_sections": sn,
            "natural_completeness": snc,
        }

    # ── Sequence aggregation ──
    sequence_agg = None
    if q_comp:
        qc = aggregate_completeness(q_comp)
        result["seq_comp_f1"] = qc["f1"]
        result["seq_comp_p"] = qc["precision"]
        result["seq_comp_r"] = qc["recall"]
        result["seq_comp_TP"] = qc["TP"]
        result["seq_comp_FP"] = qc["FP"]
        result["seq_comp_FN"] = qc["FN"]

        qp = aggregate_accuracy(q_param)
        result["seq_param_acc"] = qp["accuracy"]
        # Fix the Overall_Accuracy denominator to the GT-fixed rule-based value
        qp_gt = aggregate_accuracy(q_param_gt) if q_param_gt else qp
        result["seq_param_correct"] = qp_gt["correct"]
        result["seq_param_total"] = qp_gt["total"]

        # matched-only version
        qp_m = aggregate_accuracy(q_param_matched)
        result["seq_param_acc_matched"] = qp_m["accuracy"]
        result["seq_param_correct_matched"] = qp_m["correct"]
        result["seq_param_total_matched"] = qp_m["total"]

        qo = aggregate_order(q_order)
        result["seq_order_acc"] = qo["order_accuracy"]
        result["seq_order_correct"] = qo["correct_order"]
        result["seq_order_total"] = qo["total_blocks"]
        result["seq_violations"] = qo["constraint_violations"]

        # matched-only version
        qo_m = aggregate_order(q_order_matched)
        result["seq_order_acc_matched"] = qo_m["order_accuracy"]
        result["seq_order_correct_matched"] = qo_m["correct_order"]
        result["seq_order_total_matched"] = qo_m["total_blocks"]

        sequence_agg = {
            "stage1_completeness": qc,
            "stage2_parameter_accuracy": qp,
            "stage3_execution_order": qo,
        }

    # ── Overall Accuracy (based on GT-fixed denominator) ──
    # Sum of Inst_Param correct/total + Seq_Param correct/total
    _oa_correct = result.get("inst_param_correct", 0) + result.get("seq_param_correct", 0)
    _oa_total = result.get("inst_param_total", 0) + result.get("seq_param_total", 0)
    result["overall_accuracy"] = round(_oa_correct / _oa_total, 6) if _oa_total > 0 else 0
    result["overall_correct"] = _oa_correct
    result["overall_total"] = _oa_total

    # ── Save 5 report files (= "before" phase) ──
    if per_file_results:
        report_dir = output_dir / model_name / "before"
        report_dir.mkdir(parents=True, exist_ok=True)

        config_info = {
            "model": model_name,
            "phase": "before (v0_initial)",
            "struct_matched": len(struct_matched),
            "seq_matched": len(seq_matched),
        }
        file_matching = {
            "structured": {
                "matched": len(struct_matched),
                "unmatched_gt": struct_unmatched_gt,
                "unmatched_pred": struct_unmatched_pred,
            },
            "sequence": {
                "matched": len(seq_matched),
                "unmatched_gt": seq_unmatched_gt,
                "unmatched_pred": seq_unmatched_pred,
            },
        }
        write_reports(report_dir, config_info, file_matching,
                      per_file_results, structured_agg, sequence_agg)
        print(f"  → {report_dir}")

    # Table 3 v0 metrics (avoid duplicate evaluation of before phase)
    v0_metrics = {}
    if structured_agg:
        v0_metrics["struct_f1"] = structured_agg["stage1_completeness"]["f1"]
        v0_metrics["struct_param_acc"] = structured_agg["stage2_parameter_accuracy"]["accuracy"]
        v0_metrics["res_acc"] = structured_agg["reservoir_mapping"]["accuracy"]
        v0_metrics["nat_acc"] = structured_agg["natural_sections"]["accuracy"]
        v0_metrics["nat_comp_f1"] = structured_agg["natural_completeness"]["f1"]
        # 11 metric keys
        v0_metrics["Natural_Completeness_F1"] = structured_agg["natural_completeness"]["f1"]
        v0_metrics["Natural_Parameter_Acc"] = structured_agg["natural_sections"]["accuracy"]
        v0_metrics["Instrument_Completeness_F1"] = structured_agg["stage1_completeness"]["f1"]
        v0_metrics["Instrument_Parameter_Acc"] = structured_agg["stage2_parameter_accuracy"]["accuracy"]
        v0_metrics["Reservoir_Mapping_Acc"] = structured_agg["reservoir_mapping"]["accuracy"]
    if sequence_agg:
        v0_metrics["seq_f1"] = sequence_agg["stage1_completeness"]["f1"]
        v0_metrics["seq_param_acc"] = sequence_agg["stage2_parameter_accuracy"]["accuracy"]
        v0_metrics["seq_order_acc"] = sequence_agg["stage3_execution_order"]["order_accuracy"]
        v0_metrics["Seq_Completeness_F1"] = sequence_agg["stage1_completeness"]["f1"]
        v0_metrics["Seq_Parameter_Acc"] = sequence_agg["stage2_parameter_accuracy"]["accuracy"]
        v0_metrics["Seq_Order_Acc"] = sequence_agg["stage3_execution_order"]["order_accuracy"]
        v0_metrics["Seq_Violations"] = sequence_agg["stage3_execution_order"]["constraint_violations"]
    # Overall Accuracy (GT-fixed denominator)
    v0_metrics["Overall_Accuracy"] = result.get("overall_accuracy")
    v0_metrics["Overall_Correct"] = result.get("overall_correct")
    v0_metrics["Overall_Total"] = result.get("overall_total")
    v0_metrics["overall_acc"] = result.get("overall_accuracy")
    v0_metrics["n_structured"] = len(struct_matched)
    v0_metrics["n_sequence"] = len(seq_matched)
    v0_metrics["n_struct"] = len(struct_matched)
    v0_metrics["n_seq"] = len(seq_matched)
    result["_v0_metrics"] = v0_metrics

    return result


# ══════════════════════════════════════════════════════════
#      Validation Loop Analysis (Table 3) — per-version GT comparison

def evaluate_validation_loop(model_name: str, model_dir: Path, output_dir: Path,
                             v0_metrics: Optional[dict] = None) -> dict:
    """
    For Table 3: compute F1 + Accuracy by comparing each version against GT.
    Also generates 5 report files per (model, phase) combination.

    v0_metrics: pre-computed before-phase metrics from evaluate_model() (avoid duplicate evaluation)

    Returns:
        {
            "total_files": int,
            "versions": {
                "before":    {"struct_f1": float, "seq_f1": float, ...},
                "attempt_1": {...},
                ...
            }
        }
    """
    if not model_dir.exists():
        return {}

    # Load GT files
    gt_struct_files = _discover_files(GT_STRUCTURED_DIR, STRUCTURED_SUFFIX)
    gt_seq_files = _discover_files(GT_SEQUENCE_DIR, SEQUENCE_SUFFIX)
    gt_stems = sorted(set(gt_struct_files.keys()) & set(gt_seq_files.keys()))

    if not gt_stems:
        return {}

    # Enable LLM evaluation (use same evaluation method for all phases)
    os.environ.pop("SKIP_LLM_EVAL", None)

    version_results = {}

    for vi, label in enumerate(VERSION_LABELS):
        # before(v0) already evaluated + reports saved by evaluate_model() -> skip
        if vi == 0 and v0_metrics:
            version_results[label] = v0_metrics
            print(f"    {label}: (already evaluated -- skip)")
            continue

        per_file_results = {}
        struct_comp_list, struct_param_list = [], []
        struct_res_list, struct_natural_list, struct_nat_comp_list = [], [], []
        seq_comp_list, seq_param_list, seq_order_list = [], [], []
        seq_param_matched_list, seq_order_matched_list = [], []  # matched-only version
        # GT-fixed rule-based only — fixes the Overall_Accuracy denominator
        struct_param_gt_list, seq_param_gt_list = [], []
        struct_matched_count, seq_matched_count = 0, 0

        import time as _vtime
        _vstart = _vtime.time()
        _vtotal = len(gt_stems)

        print(f"    {label}: evaluating {_vtotal} files...")

        for _vi_idx, stem in enumerate(gt_stems):
            _vf_start = _vtime.time()
            print(f"      [{_vi_idx+1}/{_vtotal}] {stem} ...", end="", flush=True)

            # Old layout: validation/versions/   New layout: validation/{validator_model}/versions/
            _val_dir = model_dir / stem / "validation"
            version_dir = _val_dir / "versions"
            if not version_dir.exists():
                # In the new layout, search the first validator_model subfolder
                _subdirs = [d / "versions" for d in _val_dir.iterdir()
                            if d.is_dir() and (d / "versions").exists()] if _val_dir.exists() else []
                version_dir = _subdirs[0] if _subdirs else version_dir
            if not version_dir.exists():
                print(f" [SKIP] no versions dir", flush=True)
                continue

            result = {}

            # ── Structured evaluation ──
            gt_struct_path = gt_struct_files.get(stem)
            pred_struct_path = _find_best_version(version_dir, stem, vi, "structured")

            if gt_struct_path and pred_struct_path:
                try:
                    gt_text = gt_struct_path.read_text(encoding="utf-8", errors="ignore")
                    pred_text = pred_struct_path.read_text(encoding="utf-8", errors="ignore")
                    s_result = evaluate_structured(gt_text, pred_text)
                    result["structured"] = s_result
                except Exception as e:
                    result["error"] = f"Structured: {e}"

            # ── Sequence evaluation ──
            gt_seq_path = gt_seq_files.get(stem)
            pred_seq_path = _find_best_version(version_dir, stem, vi, "sequence")

            if gt_seq_path and pred_seq_path:
                try:
                    gt_df = pd.read_excel(gt_seq_path)
                    pred_df = pd.read_excel(pred_seq_path)
                    q_result = evaluate_sequence(gt_df, pred_df)
                    result["sequence"] = q_result
                except Exception as e:
                    err_msg = f"Sequence: {e}"
                    if "error" in result:
                        result["error"] += f" | {err_msg}"
                    else:
                        result["error"] = err_msg

            _vf_elapsed = _vtime.time() - _vf_start
            _vel = _vtime.time() - _vstart
            _vavg = _vel / (_vi_idx + 1)
            _vrem = _vavg * (_vtotal - _vi_idx - 1)
            _vm = int(_vrem // 60)
            _vs = int(_vrem % 60)
            _vst = "OK" if "error" not in result else "ERR"
            print(f" [{_vst}] ({_vf_elapsed:.1f}s)  remaining ~{_vm}m {_vs}s", flush=True)

            if not result:
                continue

            per_file_results[stem] = result

            if "structured" in result:
                sr = result["structured"]
                struct_comp_list.append(sr.get("llm_instrument_completeness") or sr["completeness"])
                struct_param_list.append(sr.get("llm_instrument_param_accuracy") or sr["parameter_accuracy"])
                struct_param_gt_list.append(sr["parameter_accuracy"])  # GT-fixed rule-based only
                struct_res_list.append(sr["reservoir_accuracy"])
                struct_natural_list.append(sr["natural_accuracy"])
                struct_nat_comp_list.append(sr["natural_completeness"])
                struct_matched_count += 1

            if "sequence" in result:
                qr = result["sequence"]
                seq_comp_list.append(qr.get("llm_completeness") or qr["completeness"])
                seq_param_list.append(qr.get("llm_parameter_accuracy") or qr["parameter_accuracy"])
                seq_param_gt_list.append(qr["parameter_accuracy"])  # GT-fixed rule-based only
                seq_order_list.append(qr.get("llm_execution_order") or qr["execution_order"])
                seq_param_matched_list.append(qr.get("parameter_accuracy_matched") or qr["parameter_accuracy"])
                seq_order_matched_list.append(qr.get("execution_order_matched") or qr["execution_order"])
                seq_matched_count += 1

        # ── Aggregation ──
        structured_agg = None
        if struct_comp_list:
            structured_agg = {
                "stage1_completeness": aggregate_completeness(struct_comp_list),
                "stage2_parameter_accuracy": aggregate_accuracy(struct_param_list),
                "reservoir_mapping": aggregate_accuracy(struct_res_list),
                "natural_sections": aggregate_accuracy(struct_natural_list),
                "natural_completeness": aggregate_completeness(struct_nat_comp_list),
            }

        sequence_agg = None
        sequence_agg_matched = None
        if seq_comp_list:
            sequence_agg = {
                "stage1_completeness": aggregate_completeness(seq_comp_list),
                "stage2_parameter_accuracy": aggregate_accuracy(seq_param_list),
                "stage3_execution_order": aggregate_order(seq_order_list),
            }
            sequence_agg_matched = {
                "stage2_parameter_accuracy": aggregate_accuracy(seq_param_matched_list),
                "stage3_execution_order": aggregate_order(seq_order_matched_list),
            }

        # ── Generate 5 report files ──
        if per_file_results:
            report_output_dir = output_dir / model_name / label
            report_output_dir.mkdir(parents=True, exist_ok=True)

            config_info = {
                "model": model_name,
                "phase": label,
                "version_num": vi,
                "struct_matched": struct_matched_count,
                "seq_matched": seq_matched_count,
            }
            file_matching = {
                "structured": {
                    "matched": struct_matched_count,
                    "unmatched_gt": [],
                    "unmatched_pred": [],
                },
                "sequence": {
                    "matched": seq_matched_count,
                    "unmatched_gt": [],
                    "unmatched_pred": [],
                },
            }

            write_reports(report_output_dir, config_info, file_matching,
                          per_file_results, structured_agg, sequence_agg)
            print(f"    → {report_output_dir}")

        # ── Save metrics ──
        metrics = {}
        if structured_agg:
            metrics["struct_f1"] = structured_agg["stage1_completeness"]["f1"]
            metrics["struct_param_acc"] = structured_agg["stage2_parameter_accuracy"]["accuracy"]
            metrics["res_acc"] = structured_agg["reservoir_mapping"]["accuracy"]
            metrics["nat_acc"] = structured_agg["natural_sections"]["accuracy"]
            metrics["nat_comp_f1"] = structured_agg["natural_completeness"]["f1"]
            metrics["Natural_Completeness_F1"] = structured_agg["natural_completeness"]["f1"]
            metrics["Natural_Parameter_Acc"] = structured_agg["natural_sections"]["accuracy"]
            metrics["Instrument_Completeness_F1"] = structured_agg["stage1_completeness"]["f1"]
            metrics["Instrument_Parameter_Acc"] = structured_agg["stage2_parameter_accuracy"]["accuracy"]
            metrics["Reservoir_Mapping_Acc"] = structured_agg["reservoir_mapping"]["accuracy"]
        if sequence_agg:
            metrics["seq_f1"] = sequence_agg["stage1_completeness"]["f1"]
            metrics["seq_param_acc"] = sequence_agg["stage2_parameter_accuracy"]["accuracy"]
            metrics["seq_order_acc"] = sequence_agg["stage3_execution_order"]["order_accuracy"]
            metrics["Seq_Completeness_F1"] = sequence_agg["stage1_completeness"]["f1"]
            metrics["Seq_Parameter_Acc"] = sequence_agg["stage2_parameter_accuracy"]["accuracy"]
            metrics["Seq_Parameter_Acc_Penalized"] = sequence_agg["stage2_parameter_accuracy"]["accuracy"]
            metrics["Seq_Order_Acc"] = sequence_agg["stage3_execution_order"]["order_accuracy"]
            metrics["Seq_Order_Acc_Penalized"] = sequence_agg["stage3_execution_order"]["order_accuracy"]
            metrics["Seq_Violations"] = sequence_agg["stage3_execution_order"]["constraint_violations"]
            # matched-only version
            if sequence_agg_matched:
                metrics["seq_param_acc_matched"] = sequence_agg_matched["stage2_parameter_accuracy"]["accuracy"]
                metrics["seq_order_acc_matched"] = sequence_agg_matched["stage3_execution_order"]["order_accuracy"]
                metrics["Seq_Parameter_Acc_Matched"] = sequence_agg_matched["stage2_parameter_accuracy"]["accuracy"]
                metrics["Seq_Order_Acc_Matched"] = sequence_agg_matched["stage3_execution_order"]["order_accuracy"]
        metrics["n_structured"] = struct_matched_count
        metrics["n_sequence"] = seq_matched_count
        metrics["n_struct"] = struct_matched_count
        metrics["n_seq"] = seq_matched_count
        # Overall Accuracy (GT-fixed rule-based denominator — separate from LLM metrics)
        _gt_struct_agg = aggregate_accuracy(struct_param_gt_list) if struct_param_gt_list else None
        _gt_seq_agg = aggregate_accuracy(seq_param_gt_list) if seq_param_gt_list else None
        _inst_c = _gt_struct_agg["correct"] if _gt_struct_agg else 0
        _inst_t = _gt_struct_agg["total"] if _gt_struct_agg else 0
        _seq_c = _gt_seq_agg["correct"] if _gt_seq_agg else 0
        _seq_t = _gt_seq_agg["total"] if _gt_seq_agg else 0
        _oa_c = _inst_c + _seq_c
        _oa_t = _inst_t + _seq_t
        metrics["Overall_Accuracy"] = round(_oa_c / _oa_t, 6) if _oa_t > 0 else 0
        metrics["Overall_Correct"] = _oa_c
        metrics["Overall_Total"] = _oa_t
        metrics["overall_acc"] = metrics["Overall_Accuracy"]

        version_results[label] = metrics

    # Restore environment variable
    os.environ.pop("SKIP_LLM_EVAL", None)

    return {
        "total_files": len(gt_stems),
        "versions": version_results,
    }


# ╔══════════════════════════════════════════════════════════╗
#                     Console Table Output

def print_table_1(all_results: dict, model_names: list):
    """Table 1. PDF → Structured Protocol Conversion"""
    print()
    n = all_results.get(model_names[0], {}).get("n_structured", "?") if model_names else "?"
    print("=" * 100)
    print(f"Table 1. PDF → Structured Protocol Conversion (n={n})")
    print("=" * 100)

    header = f"{'Area':<20} {'Evaluation':<22}"
    for m in model_names:
        header += f" {m:>12}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Natural [1]~[3]", "Completeness (F1)",  "nat_comp_f1"),
        ("",                 "Parameter (Acc)",    "nat_param_acc"),
        ("Instrument [5]",  "Completeness (F1)",  "inst_comp_f1"),
        ("",                "Parameter (Acc)",     "inst_param_acc"),
        ("Reservoir [4]",   "Mapping (Acc)",       "res_acc"),
    ]

    for area, evaluation, key in rows:
        line = f"{area:<20} {evaluation:<22}"
        for m in model_names:
            val = all_results.get(m, {}).get(key)
            line += f" {fmt(val):>12}"
        print(line)

    print()


def print_table_2(all_results: dict, model_names: list):
    """Table 2. Structured → Sequence Command Conversion"""
    print()
    n = all_results.get(model_names[0], {}).get("n_sequence", "?") if model_names else "?"
    print("=" * 100)
    print(f"Table 2. Structured → Sequence Command Conversion (n={n})")
    print("=" * 100)

    header = f"{'Evaluation':<25}"
    for m in model_names:
        header += f" {m:>12}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Completeness (F1)",     "seq_comp_f1"),
        ("Parameter (Acc)",       "seq_param_acc"),
        ("Execution Order (Acc)", "seq_order_acc"),
    ]

    for evaluation, key in rows:
        line = f"{evaluation:<25}"
        for m in model_names:
            val = all_results.get(m, {}).get(key)
            line += f" {fmt(val):>12}"
        print(line)

    print()


def print_table_3(all_validation: dict, model_names: list):
    """Table 3. Validation Loop Effectiveness - F1 + Accuracy"""
    print()
    print("=" * 130)
    print("Table 3. Validation Loop Effectiveness - F1 + Accuracy")
    print("=" * 130)

    header = f"{'Evaluation':<12}"
    for m in model_names:
        header += f" | {m:^40}"
    print(header)

    sub_header = f"{'':12}"
    for _ in model_names:
        sub_header += f" | {'N-F1':>8} {'S-F1':>8} {'Q-F1':>8} {'ParamAcc':>10}"
    print(sub_header)
    print("-" * len(header))

    for label in VERSION_LABELS:
        line = f"{label:<12}"
        for m in model_names:
            v = all_validation.get(m, {})
            versions = v.get("versions", {})
            vr = versions.get(label, {})
            nf1 = vr.get("nat_comp_f1")
            sf1 = vr.get("struct_f1")
            qf1 = vr.get("seq_f1")
            spa = vr.get("struct_param_acc")
            qpa = vr.get("seq_param_acc")
            # Parameter Accuracy = structured + sequence average
            if spa is not None and qpa is not None:
                param_avg = (spa + qpa) / 2
            elif spa is not None:
                param_avg = spa
            elif qpa is not None:
                param_avg = qpa
            else:
                param_avg = None

            nf1_s = f"{nf1:.4f}" if nf1 is not None else "-"
            sf1_s = f"{sf1:.4f}" if sf1 is not None else "-"
            qf1_s = f"{qf1:.4f}" if qf1 is not None else "-"
            pa_s = f"{param_avg:.4f}" if param_avg is not None else "-"
            line += f" | {nf1_s:>8} {sf1_s:>8} {qf1_s:>8} {pa_s:>10}"
        print(line)

    print()


def print_detail_table(all_results: dict, model_names: list):
    """Detailed metrics (correct/total) table"""
    print()
    print("=" * 100)
    print("Detailed metrics (correct/total)")
    print("=" * 100)

    header = f"{'Metric':<30}"
    for m in model_names:
        header += f" {m:>16}"
    print(header)
    print("-" * len(header))

    detail_rows = [
        ("[5] Param",      "inst_param_detail"),
        ("Reservoir",       "res_detail"),
        ("Natural Param",   "nat_param_detail"),
        ("Seq Param",       "seq_param_detail"),
        ("Seq Order",       "seq_order_detail"),
        ("Seq Violations",  "seq_violations"),
    ]

    for label, key in detail_rows:
        line = f"{label:<30}"
        for m in model_names:
            val = all_results.get(m, {}).get(key, "-")
            line += f" {str(val):>16}"
        print(line)

    print()


# ══════════════════════════════════════════════════════════
#                     CSV Export

def _collect_timing_and_pass_rates(model_name: str, model_dir: Path) -> dict:
    """Read the timing JSON + graph_data xlsx from the model folder to collect latency / pass rate"""
    import json as _json
    result = {}

    # ── Timing: timing_reports/*.json ──
    timing_dir = model_dir / "timing_reports"
    if timing_dir.exists():
        json_files = sorted(timing_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for jf in json_files:
            try:
                data = _json.loads(jf.read_text(encoding="utf-8"))
                totals = data.get("totals", {})
                n_files = len(data.get("files", []))
                if n_files > 0:
                    result["total_time"] = totals.get("total_sec", 0)
                    result["avg_time"] = totals.get("total_sec", 0) / n_files
                    result["avg_struct_time"] = totals.get("structuring_sec", 0) / n_files
                    result["avg_seq_time"] = totals.get("sequence_sec", 0) / n_files
                    break
            except Exception:
                pass

    # ── Pass/Fail: graph_data/*.xlsx ──
    graph_dir = model_dir / "graph_data"
    if graph_dir.exists():
        xlsx_files = sorted(graph_dir.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
        for xf in xlsx_files:
            try:
                gdf = pd.read_excel(xf)
                gdf = gdf[gdf["model"] == model_name].drop_duplicates(subset=["protocol"], keep="last")
                n = len(gdf)
                if n > 0:
                    result["n_files"] = n
                    result["pass_1st"] = int((gdf["pass_at_attempt"] == 1).sum())
                    result["pass_2nd"] = int((gdf["pass_at_attempt"] == 2).sum())
                    result["pass_3rd"] = int((gdf["pass_at_attempt"] == 3).sum())
                    result["final_fail"] = int((gdf["final_result"] == "FAIL").sum())
                    result["pass_rate"] = (result["pass_1st"] + result["pass_2nd"] + result["pass_3rd"]) / n
                    result["avg_rows"] = gdf["sequence_rows"].mean()
                    break
            except Exception:
                pass

    return result


def _save_model_comparison_csv(all_results: dict, model_names: list, output_dir: Path):
    """Table 1&2 CSV — model_comparison.csv (combined performance + latency + pass rate)"""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in model_names:
        r = all_results.get(m, {})
        if not r:
            continue

        # Collect timing + pass rate
        model_dir = RESULTS_ROOT / m
        tp = _collect_timing_and_pass_rates(m, model_dir)

        row = {
            "Model": m,
            "n_structured": r.get("n_structured", 0),
            "n_sequence": r.get("n_sequence", 0),
            # ── Structured: Natural [1]~[3] ──
            "Nat_Completeness_F1": r.get("nat_comp_f1"),
            "Nat_Precision": r.get("nat_comp_p"),
            "Nat_Recall": r.get("nat_comp_r"),
            "Nat_TP": r.get("nat_comp_TP"),
            "Nat_FP": r.get("nat_comp_FP"),
            "Nat_FN": r.get("nat_comp_FN"),
            "Nat_Parameter_Acc": r.get("nat_param_acc"),
            "Nat_Param_Correct": r.get("nat_param_correct"),
            "Nat_Param_Total": r.get("nat_param_total"),
            # ── Structured: Instrument [5] ──
            "Inst_Completeness_F1": r.get("inst_comp_f1"),
            "Inst_Precision": r.get("inst_comp_p"),
            "Inst_Recall": r.get("inst_comp_r"),
            "Inst_TP": r.get("inst_comp_TP"),
            "Inst_FP": r.get("inst_comp_FP"),
            "Inst_FN": r.get("inst_comp_FN"),
            "Inst_Parameter_Acc": r.get("inst_param_acc"),
            "Inst_Param_Correct": r.get("inst_param_correct"),
            "Inst_Param_Total": r.get("inst_param_total"),
            # ── Structured: Reservoir [4] ──
            "Reservoir_Mapping_Acc": r.get("res_acc"),
            "Reservoir_Correct": r.get("res_correct"),
            "Reservoir_Total": r.get("res_total"),
            # ── Sequence ──
            "Seq_Completeness_F1": r.get("seq_comp_f1"),
            "Seq_Precision": r.get("seq_comp_p"),
            "Seq_Recall": r.get("seq_comp_r"),
            "Seq_TP": r.get("seq_comp_TP"),
            "Seq_FP": r.get("seq_comp_FP"),
            "Seq_FN": r.get("seq_comp_FN"),
            "Seq_Parameter_Acc": r.get("seq_param_acc"),
            "Seq_Param_Correct": r.get("seq_param_correct"),
            "Seq_Param_Total": r.get("seq_param_total"),
            "Seq_Parameter_Acc_Matched": r.get("seq_param_acc_matched"),
            "Seq_Param_Correct_Matched": r.get("seq_param_correct_matched"),
            "Seq_Param_Total_Matched": r.get("seq_param_total_matched"),
            "Seq_Order_Acc": r.get("seq_order_acc"),
            "Seq_Order_Correct": r.get("seq_order_correct"),
            "Seq_Order_Total": r.get("seq_order_total"),
            "Seq_Order_Acc_Matched": r.get("seq_order_acc_matched"),
            "Seq_Order_Correct_Matched": r.get("seq_order_correct_matched"),
            "Seq_Order_Total_Matched": r.get("seq_order_total_matched"),
            "Seq_Violations": r.get("seq_violations"),
            # ── Overall Accuracy (GT-fixed denominator) ──
            "Overall_Accuracy": r.get("overall_accuracy"),
            "Overall_Correct": r.get("overall_correct"),
            "Overall_Total": r.get("overall_total"),
            # ── Timing ──
            "Avg_Time_per_File": tp.get("avg_time"),
            "Avg_Structuring_Time": tp.get("avg_struct_time"),
            "Avg_Sequence_Time": tp.get("avg_seq_time"),
            "Total_Time": tp.get("total_time"),
            # ── Validation Pass/Fail ──
            "1st_Pass": tp.get("pass_1st"),
            "2nd_Pass": tp.get("pass_2nd"),
            "3rd_Pass": tp.get("pass_3rd"),
            "Final_Fail": tp.get("final_fail"),
            "Pass_Rate": tp.get("pass_rate"),
            "Avg_Sequence_Rows": tp.get("avg_rows"),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_dir / "model_comparison.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  [CSV] {csv_path}")


def _save_validation_loop_csv(all_validation: dict, model_names: list, output_dir: Path):
    """
    Table 3 CSV — 3 files:
      1) validation_loop_metrics.csv (backward compatible)
      2) validation_loop_metrics_full.csv (before + final attempt, 11 metrics)
      3) validation_loop_metrics_by_phase.csv (all phases x models, 11 metrics)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) Legacy CSV (backward compatible) ──
    headers = ["Evaluation"]
    for m in model_names:
        headers.extend([
            f"{m}_nat_comp_f1",
            f"{m}_struct_f1",
            f"{m}_seq_f1",
            f"{m}_nat_acc",
            f"{m}_struct_param_acc",
            f"{m}_seq_param_acc",
            f"{m}_res_acc",
            f"{m}_seq_order_acc",
            f"{m}_overall_acc",
        ])

    rows = []
    for label in VERSION_LABELS:
        row = {"Evaluation": label}
        for m in model_names:
            v = all_validation.get(m, {})
            versions = v.get("versions", {})
            vr = versions.get(label, {})
            for key in ["nat_comp_f1", "struct_f1", "seq_f1", "nat_acc",
                         "struct_param_acc", "seq_param_acc", "res_acc", "seq_order_acc",
                         "overall_acc"]:
                val = vr.get(key)
                row[f"{m}_{key}"] = f"{val:.4f}" if val is not None else "-"
        rows.append(row)

    csv_path = output_dir / "validation_loop_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] {csv_path}")

    # ── 2) Extended CSV: before + final attempt, 11 metrics ──
    full_headers = [
        "Phase", "Model", "n_structured", "n_sequence",
        "Natural_Completeness_F1", "Natural_Parameter_Acc",
        "Instrument_Completeness_F1", "Instrument_Parameter_Acc",
        "Reservoir_Mapping_Acc",
        "Seq_Completeness_F1", "Seq_Parameter_Acc",
        "Seq_Order_Acc", "Seq_Violations",
        "Overall_Accuracy", "Overall_Correct", "Overall_Total",
    ]

    def _make_full_row(phase_label, model, m):
        return {
            "Phase": phase_label,
            "Model": model,
            "n_structured": _fmt_csv(m.get("n_structured")),
            "n_sequence": _fmt_csv(m.get("n_sequence")),
            "Natural_Completeness_F1": _fmt_csv(m.get("Natural_Completeness_F1") or m.get("nat_comp_f1")),
            "Natural_Parameter_Acc": _fmt_csv(m.get("Natural_Parameter_Acc") or m.get("nat_acc")),
            "Instrument_Completeness_F1": _fmt_csv(m.get("Instrument_Completeness_F1") or m.get("struct_f1")),
            "Instrument_Parameter_Acc": _fmt_csv(m.get("Instrument_Parameter_Acc") or m.get("struct_param_acc")),
            "Reservoir_Mapping_Acc": _fmt_csv(m.get("Reservoir_Mapping_Acc") or m.get("res_acc")),
            "Seq_Completeness_F1": _fmt_csv(m.get("Seq_Completeness_F1") or m.get("seq_f1")),
            "Seq_Parameter_Acc": _fmt_csv(m.get("Seq_Parameter_Acc") or m.get("seq_param_acc")),
            "Seq_Order_Acc": _fmt_csv(m.get("Seq_Order_Acc") or m.get("seq_order_acc")),
            "Seq_Violations": _fmt_csv(m.get("Seq_Violations")),
            "Overall_Accuracy": _fmt_csv(m.get("Overall_Accuracy")),
            "Overall_Correct": _fmt_csv(m.get("Overall_Correct")),
            "Overall_Total": _fmt_csv(m.get("Overall_Total")),
        }

    full_rows = []
    for model in model_names:
        v = all_validation.get(model, {})
        versions = v.get("versions", {})

        # before
        pre = versions.get("before")
        if pre:
            full_rows.append(_make_full_row("before", model, pre))

        # final attempt (attempt_3 > attempt_2 > attempt_1)
        best_metrics = None
        best_phase = None
        for label in reversed(VERSION_LABELS):
            if label == "before":
                continue
            m = versions.get(label)
            if m:
                best_metrics = m
                best_phase = label
                break
        if best_metrics:
            full_rows.append(_make_full_row(best_phase, model, best_metrics))

    full_csv_path = output_dir / "validation_loop_metrics_full.csv"
    with open(full_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=full_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(full_rows)
    print(f"  [CSV] {full_csv_path}")

    # ── 3) Per-phase x per-model full CSV ──
    phase_rows = []
    for label in VERSION_LABELS:
        for model in model_names:
            v = all_validation.get(model, {})
            versions = v.get("versions", {})
            m = versions.get(label)
            if m:
                phase_rows.append(_make_full_row(label, model, m))

    phase_csv_path = output_dir / "validation_loop_metrics_by_phase.csv"
    with open(phase_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=full_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(phase_rows)
    print(f"  [CSV] {phase_csv_path}")


def _save_validation_by_phase_csv(all_validation: dict, model_names: list, output_dir: Path):
    """
    validation_by_phase.csv — Phase x Model (4xN rows), 11 metrics.
    For Table 3 + Fig 3 (line chart).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        "Phase", "Model", "n_structured", "n_sequence",
        "Natural_Completeness_F1", "Natural_Parameter_Acc",
        "Instrument_Completeness_F1", "Instrument_Parameter_Acc",
        "Reservoir_Mapping_Acc",
        "Seq_Completeness_F1", "Seq_Parameter_Acc",
        "Seq_Order_Acc", "Seq_Violations",
        "Overall_Accuracy", "Overall_Correct", "Overall_Total",
    ]

    rows = []
    for label in VERSION_LABELS:
        for model in model_names:
            v = all_validation.get(model, {})
            versions = v.get("versions", {})
            m = versions.get(label)
            if not m:
                continue
            rows.append({
                "Phase": label,
                "Model": model,
                "n_structured": _fmt_csv(m.get("n_structured") or m.get("n_struct")),
                "n_sequence": _fmt_csv(m.get("n_sequence") or m.get("n_seq")),
                "Natural_Completeness_F1": _fmt_csv(m.get("Natural_Completeness_F1") or m.get("nat_comp_f1")),
                "Natural_Parameter_Acc": _fmt_csv(m.get("Natural_Parameter_Acc") or m.get("nat_acc")),
                "Instrument_Completeness_F1": _fmt_csv(m.get("Instrument_Completeness_F1") or m.get("struct_f1")),
                "Instrument_Parameter_Acc": _fmt_csv(m.get("Instrument_Parameter_Acc") or m.get("struct_param_acc")),
                "Reservoir_Mapping_Acc": _fmt_csv(m.get("Reservoir_Mapping_Acc") or m.get("res_acc")),
                "Seq_Completeness_F1": _fmt_csv(m.get("Seq_Completeness_F1") or m.get("seq_f1")),
                "Seq_Parameter_Acc": _fmt_csv(m.get("Seq_Parameter_Acc") or m.get("seq_param_acc")),
                "Seq_Order_Acc": _fmt_csv(m.get("Seq_Order_Acc") or m.get("seq_order_acc")),
                "Seq_Violations": _fmt_csv(m.get("Seq_Violations")),
                "Overall_Accuracy": _fmt_csv(m.get("Overall_Accuracy")),
                "Overall_Correct": _fmt_csv(m.get("Overall_Correct")),
                "Overall_Total": _fmt_csv(m.get("Overall_Total")),
            })

    csv_path = output_dir / "validation_by_phase.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] {csv_path}")


def _save_before_vs_after_csv(all_validation: dict, model_names: list, output_dir: Path):
    """
    before_vs_after.csv — N model rows, before/final/delta x core metrics.
    For Fig 2 (Grouped Bar Chart).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        ("Natural_Completeness_F1", "nat_comp_f1"),
        ("Natural_Parameter_Acc", "nat_acc"),
        ("Instrument_Completeness_F1", "struct_f1"),
        ("Instrument_Parameter_Acc", "struct_param_acc"),
        ("Reservoir_Mapping_Acc", "res_acc"),
        ("Seq_Completeness_F1", "seq_f1"),
        ("Seq_Parameter_Acc", "seq_param_acc"),
        ("Seq_Order_Acc", "seq_order_acc"),
        ("Seq_Violations", None),
        ("Overall_Accuracy", "overall_acc"),
    ]

    headers = ["Model"]
    for name, _ in metric_keys:
        headers.extend([f"{name}_before", f"{name}_after", f"{name}_delta"])

    rows = []
    for model in model_names:
        v = all_validation.get(model, {})
        versions = v.get("versions", {})

        # before
        before = versions.get("before", {})

        # final (attempt_3 > attempt_2 > attempt_1)
        after = {}
        for label in reversed(VERSION_LABELS):
            if label == "before":
                continue
            m = versions.get(label)
            if m:
                after = m
                break

        row = {"Model": model}
        for name, alt_key in metric_keys:
            bv = before.get(name) or (before.get(alt_key) if alt_key else None)
            av = after.get(name) or (after.get(alt_key) if alt_key else None)

            row[f"{name}_before"] = _fmt_csv(bv)
            row[f"{name}_after"] = _fmt_csv(av)

            if bv is not None and av is not None:
                try:
                    delta = float(av) - float(bv)
                    row[f"{name}_delta"] = round(delta, 4)
                except (TypeError, ValueError):
                    row[f"{name}_delta"] = ""
            else:
                row[f"{name}_delta"] = ""

        rows.append(row)

    csv_path = output_dir / "before_vs_after.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] {csv_path}")


def _save_model_profile_csv(all_validation: dict, model_names: list, output_dir: Path):
    """
    model_profile.csv — N model rows, full metrics (final values).
    For Fig 1 (Radar Chart).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        "Model",
        "Natural_Completeness_F1", "Natural_Parameter_Acc",
        "Instrument_Completeness_F1", "Instrument_Parameter_Acc",
        "Reservoir_Mapping_Acc",
        "Seq_Completeness_F1", "Seq_Parameter_Acc",
        "Seq_Order_Acc", "Seq_Violations",
        "Overall_Accuracy",
    ]

    alt_map = {
        "Natural_Completeness_F1": "nat_comp_f1",
        "Natural_Parameter_Acc": "nat_acc",
        "Instrument_Completeness_F1": "struct_f1",
        "Instrument_Parameter_Acc": "struct_param_acc",
        "Reservoir_Mapping_Acc": "res_acc",
        "Seq_Completeness_F1": "seq_f1",
        "Seq_Parameter_Acc": "seq_param_acc",
        "Seq_Order_Acc": "seq_order_acc",
        "Seq_Violations": None,
        "Overall_Accuracy": "overall_acc",
    }

    rows = []
    for model in model_names:
        v = all_validation.get(model, {})
        versions = v.get("versions", {})

        # Final attempt (attempt_3 > attempt_2 > attempt_1 > before)
        final = {}
        for label in reversed(VERSION_LABELS):
            m = versions.get(label)
            if m:
                final = m
                break

        row = {"Model": model}
        for h in headers[1:]:
            alt = alt_map.get(h)
            val = final.get(h) or (final.get(alt) if alt else None)
            row[h] = _fmt_csv(val)
        rows.append(row)

    csv_path = output_dir / "model_profile.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] {csv_path}")


def _save_full_unified_table_csv(all_validation: dict, model_names: list, output_dir: Path):
    """
    full_unified_table.csv — Phase x Area x Evaluation x Models unified table.
    6 phases (before, attempt_1~3, final, delta) x 9 metrics x N models.
    For paper raw data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 9 row definitions: (Area, Evaluation, metric_primary_key, metric_alt_key)
    METRIC_ROWS = [
        ("Natural [1]~[3]", "Completeness (F1)",  "Natural_Completeness_F1",    "nat_comp_f1"),
        ("",                 "Parameter (Acc)",    "Natural_Parameter_Acc",      "nat_acc"),
        ("Instrument [5]",  "Completeness (F1)",  "Instrument_Completeness_F1", "struct_f1"),
        ("",                "Parameter (Acc)",     "Instrument_Parameter_Acc",   "struct_param_acc"),
        ("Reservoir [4]",   "Mapping (Acc)",       "Reservoir_Mapping_Acc",      "res_acc"),
        ("Sequence",        "Completeness (F1)",          "Seq_Completeness_F1",        "seq_f1"),
        ("",                "Parameter (Acc) [Penalized]", "Seq_Parameter_Acc_Penalized", "seq_param_acc"),
        ("",                "Parameter (Acc) [Matched]",   "Seq_Parameter_Acc_Matched",   "seq_param_acc_matched"),
        ("",                "Order (Acc) [Penalized]",     "Seq_Order_Acc_Penalized",     "seq_order_acc"),
        ("",                "Order (Acc) [Matched]",       "Seq_Order_Acc_Matched",       "seq_order_acc_matched"),
        ("",                "Violations",                   "Seq_Violations",              None),
        ("Overall",         "Accuracy",                     "Overall_Accuracy",            "overall_acc"),
    ]

    # Phase definitions: before, attempt_1~3, final, delta
    PHASES = ["before", "attempt_1", "attempt_2", "attempt_3", "final", "delta"]

    # Build per-model phase-level metric dict
    def _get_version_metrics(model):
        v = all_validation.get(model, {})
        return v.get("versions", {})

    def _get_final(versions):
        """Final attempt (attempt_3 > attempt_2 > attempt_1)"""
        for label in reversed(VERSION_LABELS):
            if label == "before":
                continue
            m = versions.get(label)
            if m:
                return m
        return {}

    def _get_val(metrics_dict, primary, alt):
        if not metrics_dict:
            return None
        v = metrics_dict.get(primary)
        if v is not None:
            return v
        if alt:
            return metrics_dict.get(alt)
        return None

    # CSV headers
    headers = ["Phase", "Area", "Evaluation"] + model_names

    rows = []
    for phase in PHASES:
        for area, evaluation, pk, ak in METRIC_ROWS:
            row = {
                "Phase": phase,
                "Area": area,
                "Evaluation": evaluation,
            }

            for model in model_names:
                versions = _get_version_metrics(model)

                if phase == "final":
                    m = _get_final(versions)
                    val = _get_val(m, pk, ak)
                elif phase == "delta":
                    before_m = versions.get("before", {})
                    after_m = _get_final(versions)
                    bv = _get_val(before_m, pk, ak)
                    av = _get_val(after_m, pk, ak)
                    if bv is not None and av is not None:
                        try:
                            val = round(float(av) - float(bv), 4)
                        except (TypeError, ValueError):
                            val = None
                    else:
                        val = None
                else:
                    m = versions.get(phase, {})
                    val = _get_val(m, pk, ak)

                if val is not None:
                    if isinstance(val, float):
                        row[model] = round(val, 4)
                    else:
                        row[model] = val
                else:
                    row[model] = ""

            rows.append(row)

    csv_path = output_dir / "full_unified_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] {csv_path}")


def _write_aggregated_error_log(output_dir: Path, model_names: list):
    """
    Aggregate evaluation_errors_only.txt from all models
    into a single aggregated_errors.txt.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Aggregated Error Log -- All Models x All Phases")
    lines.append("=" * 80)

    for model in model_names:
        model_dir = output_dir / model
        if not model_dir.exists():
            continue

        for label in VERSION_LABELS:
            err_file = model_dir / label / "evaluation_errors_only.txt"
            if not err_file.exists():
                continue

            content = err_file.read_text(encoding="utf-8", errors="ignore").strip()
            if not content or content == "(No errors)":
                continue

            lines.append("")
            lines.append(f"{'─'*60}")
            lines.append(f"[{model}] [{label}]")
            lines.append(f"{'─'*60}")
            lines.append(content)

    if len(lines) <= 3:
        lines.append("\n(No errors -- all models all phases normal)")

    agg_path = output_dir / "aggregated_errors.txt"
    with open(agg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Report] {agg_path}")


# ══════════════════════════════════════════════════════════
#                     Main Execution

def main():
    """Auto-discover all models -> evaluate -> print tables -> save CSV"""

    # ── Auto-discover models ──
    model_dirs = _auto_discover_model_dirs()

    if not model_dirs:
        print(f"[ERROR] No model results folders found: {RESULTS_ROOT}")
        return

    model_names = list(model_dirs.keys())

    print("=" * 60)
    print("Full model comparison evaluation")
    print(f"Detected models: {', '.join(model_names)}")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    all_results = {}      # for Table 1&2
    all_validation = {}   # for Table 3

    for model_name in model_names:
        model_dir = model_dirs[model_name]

        print(f"\n{'─' * 50}")
        print(f"[{model_name}] Evaluating...")
        print(f"{'─' * 50}")

        # ── Table 1 & 2: v0_initial performance evaluation + 5 reports ──
        result = evaluate_model(model_name, model_dir, OUTPUT_DIR)
        if result:
            all_results[model_name] = result
            print(f"  Structured: {result.get('n_structured', 0)} files")
            print(f"  Sequence:   {result.get('n_sequence', 0)} files")

        # ── Table 3: validation loop (GT comparison F1+Accuracy) + 5 reports per phase ──
        v0_metrics = result.get("_v0_metrics") if result else None
        validation = evaluate_validation_loop(model_name, model_dir, OUTPUT_DIR,
                                              v0_metrics=v0_metrics)
        if validation:
            all_validation[model_name] = validation
            print(f"  Validation: {validation.get('total_files', 0)} files")
            for lbl, vr in validation.get("versions", {}).items():
                sf1 = vr.get("struct_f1")
                qf1 = vr.get("seq_f1")
                if sf1 is not None:
                    print(f"    {lbl}: S-F1={sf1:.4f}  Q-F1={qf1:.4f}")

        # ── Save CSV immediately after each model completes (incremental) ──
        _save_model_comparison_csv(all_results, model_names, OUTPUT_DIR)
        _save_validation_loop_csv(all_validation, model_names, OUTPUT_DIR)

    # ── Save 4 paper CSVs ──
    _save_validation_by_phase_csv(all_validation, model_names, OUTPUT_DIR)
    _save_before_vs_after_csv(all_validation, model_names, OUTPUT_DIR)
    _save_model_profile_csv(all_validation, model_names, OUTPUT_DIR)
    _save_full_unified_table_csv(all_validation, model_names, OUTPUT_DIR)

    # ── Aggregated error log ──
    _write_aggregated_error_log(OUTPUT_DIR, model_names)

    # ── Final console table output ──
    print_table_1(all_results, model_names)
    print_table_2(all_results, model_names)
    print_table_3(all_validation, model_names)
    print_detail_table(all_results, model_names)

    # ── Summary Table (Timing + Pass Rate + Key Metrics) ──
    print()
    print("=" * 160)
    print("Summary: Performance + Timing + Validation Pass Rate")
    print("=" * 160)
    hdr = f"{'Model':25s} {'Seq_F1':>8s} {'Param_P':>8s} {'Param_M':>8s} {'Order':>8s} {'Viol':>5s} {'AvgTime':>8s} {'Struct':>8s} {'Seq':>8s} {'1st':>4s} {'2nd':>4s} {'3rd':>4s} {'FAIL':>5s} {'Pass%':>6s} {'Rows':>6s}"
    print(hdr)
    print("-" * 160)
    for m in model_names:
        r = all_results.get(m, {})
        tp = _collect_timing_and_pass_rates(m, RESULTS_ROOT / m)
        sf1 = r.get("seq_comp_f1")
        sp = r.get("seq_param_acc")
        sm = r.get("seq_param_acc_matched")
        so = r.get("seq_order_acc")
        sv = r.get("seq_violations")
        at = tp.get("avg_time")
        ast = tp.get("avg_struct_time")
        ase = tp.get("avg_seq_time")
        p1 = tp.get("pass_1st")
        p2 = tp.get("pass_2nd")
        p3 = tp.get("pass_3rd")
        ff = tp.get("final_fail")
        pr = tp.get("pass_rate")
        ar = tp.get("avg_rows")

        def _f(v, d=4):
            return f"{v:.{d}f}" if v is not None else "-"
        def _i(v):
            return f"{v}" if v is not None else "-"
        def _t(v):
            return f"{v:.1f}s" if v is not None else "-"
        def _p(v):
            return f"{v:.0%}" if v is not None else "-"

        line = f"{m:25s} {_f(sf1):>8s} {_f(sp):>8s} {_f(sm):>8s} {_f(so):>8s} {_i(sv):>5s} {_t(at):>8s} {_t(ast):>8s} {_t(ase):>8s} {_i(p1):>4s} {_i(p2):>4s} {_i(p3):>4s} {_i(ff):>5s} {_p(pr):>6s} {_f(ar,0):>6s}"
        print(line)
    print("=" * 160)

    # ── Completion message ──
    print(f"\n{'='*70}")
    print(f"All done!")
    print(f"  Report root:           {OUTPUT_DIR}")
    print(f"  Model comparison CSV:  {OUTPUT_DIR / 'model_comparison.csv'}")
    print(f"  Validation loop CSV:   {OUTPUT_DIR / 'validation_loop_metrics.csv'}")
    print(f"  Full metrics CSV:      {OUTPUT_DIR / 'validation_loop_metrics_full.csv'}")
    print(f"  Per-phase metrics CSV: {OUTPUT_DIR / 'validation_loop_metrics_by_phase.csv'}")
    print(f"  Paper phase CSV:       {OUTPUT_DIR / 'validation_by_phase.csv'}")
    print(f"  Paper before/after CSV:{OUTPUT_DIR / 'before_vs_after.csv'}")
    print(f"  Paper profile CSV:     {OUTPUT_DIR / 'model_profile.csv'}")
    print(f"  Paper unified CSV:     {OUTPUT_DIR / 'full_unified_table.csv'}")
    print(f"  Aggregated error log:  {OUTPUT_DIR / 'aggregated_errors.txt'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
