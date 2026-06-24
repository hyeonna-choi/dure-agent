"""
report_writer.py — Generate JSON, TXT, CSV reports + Confusion Matrix image.
"""

import json
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # Save images without GUI
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


def write_reports(
    output_dir: Path,
    config_info: Dict,
    file_matching: Dict,
    per_file_results: Dict[str, Dict],
    structured_agg: Optional[Dict],
    sequence_agg: Optional[Dict],
):
    """
    Generate 3 report files:
    - evaluation_summary.json
    - evaluation_details.txt
    - evaluation_per_file.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir, config_info, file_matching, per_file_results,
                structured_agg, sequence_agg)
    _write_details_txt(output_dir, per_file_results)
    _write_csv(output_dir, per_file_results)
    _write_errors_only(output_dir, per_file_results)
    _write_confusion_matrix(output_dir, per_file_results)


def _write_json(output_dir, config_info, file_matching, per_file_results,
                structured_agg, sequence_agg):
    """evaluation_summary.json"""
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config_info,
        "file_matching": file_matching,
    }

    if structured_agg:
        summary["structured"] = structured_agg
    if sequence_agg:
        summary["sequence"] = sequence_agg

    # Overall Accuracy (fixed GT denominator, includes FP penalty)
    _s_correct = structured_agg["stage2_parameter_accuracy"]["correct"] if structured_agg else 0
    _s_total = structured_agg["stage2_parameter_accuracy"]["total"] if structured_agg else 0
    _q_correct = sequence_agg["stage2_parameter_accuracy"]["correct"] if sequence_agg else 0
    _q_total = sequence_agg["stage2_parameter_accuracy"]["total"] if sequence_agg else 0
    _oa_correct = _s_correct + _q_correct
    _oa_total = _s_total + _q_total
    summary["overall"] = {
        "accuracy": round(_oa_correct / _oa_total, 6) if _oa_total > 0 else 0,
        "correct": _oa_correct,
        "total": _oa_total,
    }

    # Per-file summary (details go to txt)
    per_file_summary = {}
    for stem, result in per_file_results.items():
        entry = {}
        if "structured" in result:
            s = result["structured"]
            entry["structured"] = {
                "completeness": s["completeness"].to_dict(),
                "parameter_accuracy": s["parameter_accuracy"].to_dict(),
                "reservoir_accuracy": s["reservoir_accuracy"].to_dict(),
                "natural_completeness": s["natural_completeness"].to_dict(),
                "natural_accuracy": s["natural_accuracy"].to_dict(),
            }
            # LLM [5] results
            if s.get("llm_instrument_completeness"):
                entry["structured"]["llm_instrument_completeness"] = s["llm_instrument_completeness"].to_dict()
            if s.get("llm_instrument_param_accuracy"):
                entry["structured"]["llm_instrument_param_accuracy"] = s["llm_instrument_param_accuracy"].to_dict()
            # Store detail data for post-processing
            if s.get("reservoir_detail"):
                entry["structured"]["reservoir_detail"] = s["reservoir_detail"]
            if s.get("completeness_detail"):
                entry["structured"]["completeness_detail"] = s["completeness_detail"]
            if s.get("parameter_detail"):
                entry["structured"]["parameter_detail"] = s["parameter_detail"]
            if s.get("natural_detail"):
                entry["structured"]["natural_detail"] = s["natural_detail"]
            if s.get("llm_instrument_detail"):
                entry["structured"]["llm_instrument_detail"] = s["llm_instrument_detail"]
        if "sequence" in result:
            q = result["sequence"]
            entry["sequence"] = {
                "completeness": q["completeness"].to_dict(),
                "parameter_accuracy": q["parameter_accuracy"].to_dict(),
                "execution_order": q["execution_order"].to_dict(),
            }
            # LLM sequence results
            if q.get("llm_completeness"):
                entry["sequence"]["llm_completeness"] = q["llm_completeness"].to_dict()
            if q.get("llm_parameter_accuracy"):
                entry["sequence"]["llm_parameter_accuracy"] = q["llm_parameter_accuracy"].to_dict()
            if q.get("llm_execution_order"):
                entry["sequence"]["llm_execution_order"] = q["llm_execution_order"].to_dict()
            # Store detail data for post-processing
            if q.get("completeness_detail"):
                entry["sequence"]["completeness_detail"] = q["completeness_detail"]
            if q.get("parameter_detail"):
                entry["sequence"]["parameter_detail"] = q["parameter_detail"]
            if q.get("llm_detail"):
                entry["sequence"]["llm_detail"] = q["llm_detail"]
        if "error" in result:
            entry["error"] = result["error"]
        per_file_summary[stem] = entry

    summary["per_file"] = per_file_summary

    path = output_dir / "evaluation_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Report] {path}")


def _write_details_txt(output_dir, per_file_results):
    """evaluation_details.txt — Per-file detailed error report"""
    lines = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Performance Evaluation Details — {ts}")
    lines.append("=" * 80)

    for stem in sorted(per_file_results.keys()):
        result = per_file_results[stem]
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"[FILE] {stem}")
        lines.append("=" * 80)

        if "error" in result:
            lines.append(f"  ERROR: {result['error']}")
            continue

        # ── Structured ──
        if "structured" in result:
            s = result["structured"]
            lines.append("")
            lines.append("--- STRUCTURED PROTOCOL ---")
            lines.append("")

            # Stage 1
            comp = s["completeness"]
            cd = s["completeness_detail"]
            lines.append(f"[Stage 1: Completeness]  GT={cd['gt_action_count']}  Pred={cd['pred_action_count']}")
            lines.append(f"  TP ({comp.TP}):")
            for tp in cd["tp"]:
                lines.append(f"    #{tp['gt_idx']+1} {tp['gt_desc']} <-> {tp['pred_desc']} OK")
            if cd["fn"]:
                lines.append(f"  FN ({comp.FN}) — Missing:")
                for fn in cd["fn"]:
                    lines.append(f"    #{fn['gt_idx']+1} {fn['gt_desc']}  line: {fn.get('gt_line','')}")
            else:
                lines.append(f"  FN (0): (None)")
            if cd["fp"]:
                lines.append(f"  FP ({comp.FP}) — Extra (FP):")
                for fp in cd["fp"]:
                    lines.append(f"    #{fp['pred_idx']+1} {fp['pred_desc']}  line: {fp.get('pred_line','')}")
            else:
                lines.append(f"  FP (0): (None)")
            lines.append(f"  Precision={comp.precision:.3f}  Recall={comp.recall:.3f}  F1={comp.f1:.3f}")

            # Stage 2
            pa = s["parameter_accuracy"]
            lines.append(f"")
            lines.append(f"[Stage 2: Parameter Accuracy]")
            for d in s["parameter_detail"]:
                mark = "OK" if d["match"] else "WRONG"
                lines.append(f"  {d['gt_action']} {d['param']}: GT={d['gt_val']}  Pred={d['pred_val']}  {mark}")
            lines.append(f"  Accuracy: {pa.correct}/{pa.total} = {pa.accuracy:.3f}")

            # Reservoir
            ra = s["reservoir_accuracy"]
            lines.append(f"")
            lines.append(f"[Reservoir Mapping]")
            res_detail = s["reservoir_detail"]
            if res_detail:
                for d in res_detail:
                    mark = "OK" if d.get("match", False) else "WRONG"
                    reason = f"  ({d['reason']})" if d.get("reason") else ""
                    lines.append(f"  R{d['reservoir']}: GT={d['gt_reagent']}  |  Pred={d['pred_reagent']}  {mark}{reason}")
            lines.append(f"  {ra.correct}/{ra.total} = {ra.accuracy:.3f}")

            # Natural [1]~[3]
            nc = s.get("natural_completeness")
            na = s["natural_accuracy"]
            nd = s.get("natural_detail", [])
            lines.append(f"")
            if nc:
                lines.append(f"[Natural Sections [1]~[3] Completeness]")
                lines.append(f"  TP={nc.TP} FP={nc.FP} FN={nc.FN}  F1={nc.f1:.4f}")
            lines.append(f"[Natural Sections [1]~[3] Parameter Accuracy]")
            if nd:
                for d in nd:
                    lines.append(f"  {d['section']}")
                    if "missing_from_pred" in d:
                        lines.append(f"    Missing (only in GT):")
                        for item in d["missing_from_pred"]:
                            lines.append(f"      - {item}")
                        if "missing_source_lines" in d:
                            lines.append(f"    GT original line:")
                            for sl in d["missing_source_lines"]:
                                lines.append(f"      {sl}")
                    if "extra_in_pred" in d:
                        lines.append(f"    Extra (FP) (only in Pred):")
                        for item in d["extra_in_pred"]:
                            lines.append(f"      + {item}")
                        if "extra_source_lines" in d:
                            lines.append(f"    Pred original line:")
                            for sl in d["extra_source_lines"]:
                                lines.append(f"      {sl}")
            else:
                lines.append(f"  (No differences)")
            lines.append(f"  {na.correct}/{na.total} = {na.accuracy:.3f}")

            # ── LLM-based [5] evaluation results ──
            llm_ic = s.get("llm_instrument_completeness")
            llm_ip = s.get("llm_instrument_param_accuracy")
            llm_id = s.get("llm_instrument_detail", [])
            if llm_ic:
                lines.append(f"")
                lines.append(f"[LLM-based [5] Instrument Evaluation]")
                lines.append(f"  Completeness: TP={llm_ic.TP}  FN={llm_ic.FN}  FP={llm_ic.FP}  F1={llm_ic.f1:.3f}")
                if llm_ip:
                    lines.append(f"  Param Accuracy: {llm_ip.correct}/{llm_ip.total} = {llm_ip.accuracy:.3f}")
                for d in llm_id:
                    if "fn_actions" in d:
                        lines.append(f"  Missing (FN):")
                        for item in d["fn_actions"]:
                            lines.append(f"    - {item}")
                    if "fp_actions" in d:
                        lines.append(f"  Extra (FP):")
                        for item in d["fp_actions"]:
                            lines.append(f"    + {item}")
                    if "wrong_params" in d:
                        lines.append(f"  Parameter error:")
                        for item in d["wrong_params"]:
                            lines.append(f"    ! {item}")

        # ── Sequence ──
        if "sequence" in result:
            q = result["sequence"]
            lines.append("")
            lines.append("--- SEQUENCE ---")
            lines.append("")

            # Stage 1
            comp = q["completeness"]
            cd = q["completeness_detail"]
            lines.append(f"[Stage 1: Completeness]  GT blocks={cd['gt_block_count']}  Pred blocks={cd['pred_block_count']}")
            lines.append(f"  TP ({comp.TP}):")
            for tp in cd["tp"]:
                lines.append(f"    GT[{tp['gt_rows']}] {tp['gt_desc']} <-> Pred[{tp['pred_rows']}] {tp['pred_desc']} OK")
            if cd["fn"]:
                lines.append(f"  FN ({comp.FN}) — Missing:")
                for fn in cd["fn"]:
                    lines.append(f"    GT[{fn['gt_rows']}] {fn['gt_desc']}")
            else:
                lines.append(f"  FN (0): (None)")
            if cd["fp"]:
                lines.append(f"  FP ({comp.FP}) — Extra (FP):")
                for fp in cd["fp"]:
                    lines.append(f"    Pred[{fp['pred_rows']}] {fp['pred_desc']}")
            else:
                lines.append(f"  FP (0): (None)")
            lines.append(f"  Precision={comp.precision:.3f}  Recall={comp.recall:.3f}  F1={comp.f1:.3f}")

            # Stage 2
            pa = q["parameter_accuracy"]
            lines.append(f"")
            lines.append(f"[Stage 2: Parameter Accuracy]")
            for d in q["parameter_detail"]:
                mark = "OK" if d["match"] else "WRONG"
                lines.append(f"  {d['gt_block']} {d['param']}: GT={d['gt_val']}  Pred={d['pred_val']}  {mark}")
            lines.append(f"  Accuracy: {pa.correct}/{pa.total} = {pa.accuracy:.3f}")

            # Stage 3
            eo = q["execution_order"]
            lines.append(f"")
            lines.append(f"[Stage 3: Execution Order]")
            lines.append(f"  Block order match: {eo.correct_order}/{eo.total_blocks} = {eo.order_accuracy:.3f}")
            lines.append(f"  Constraint violations: {eo.constraint_violations}")

            # ── LLM-based sequence evaluation results ──
            llm_sc = q.get("llm_completeness")
            llm_sp = q.get("llm_parameter_accuracy")
            llm_so = q.get("llm_execution_order")
            llm_sd = q.get("llm_detail", [])
            if llm_sc:
                lines.append(f"")
                lines.append(f"[LLM-based Sequence Evaluation]")
                lines.append(f"  Completeness: TP={llm_sc.TP}  FN={llm_sc.FN}  FP={llm_sc.FP}  F1={llm_sc.f1:.3f}")
                if llm_sp:
                    lines.append(f"  Param Accuracy: {llm_sp.correct}/{llm_sp.total} = {llm_sp.accuracy:.3f}")
                if llm_so:
                    lines.append(f"  Order Accuracy: {llm_so.correct_order}/{llm_so.total_blocks} = {llm_so.order_accuracy:.3f}")
                for d in llm_sd:
                    if "fn_blocks" in d:
                        lines.append(f"  Missing (FN):")
                        for item in d["fn_blocks"]:
                            lines.append(f"    - {item}")
                    if "fp_blocks" in d:
                        lines.append(f"  Extra (FP):")
                        for item in d["fp_blocks"]:
                            lines.append(f"    + {item}")
                    if "wrong_params" in d:
                        lines.append(f"  Parameter error:")
                        for item in d["wrong_params"]:
                            lines.append(f"    ! {item}")
                    if "order_errors" in d:
                        lines.append(f"  Order error:")
                        for item in d["order_errors"]:
                            lines.append(f"    ! {item}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")

    path = output_dir / "evaluation_details.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] {path}")


def _write_csv(output_dir, per_file_results):
    """evaluation_per_file.csv"""
    headers = [
        "file",
        # Rule-based Structured
        "s_TP", "s_FP", "s_FN", "s_TN", "s_prec", "s_rec", "s_f1",
        "s_param_correct", "s_param_wrong", "s_param_total", "s_param_acc",
        "s_res_correct", "s_res_total", "s_res_acc",
        "s_nat_correct", "s_nat_total", "s_nat_acc",
        # LLM-based [5]
        "s_llm_TP", "s_llm_FP", "s_llm_FN", "s_llm_f1",
        "s_llm_param_correct", "s_llm_param_total", "s_llm_param_acc",
        # Rule-based Sequence (penalized = default)
        "q_TP", "q_FP", "q_FN", "q_TN", "q_prec", "q_rec", "q_f1",
        "q_param_correct", "q_param_wrong", "q_param_total", "q_param_acc",
        "q_order_correct", "q_order_total", "q_order_acc", "q_violations",
        # Matched-only versions
        "q_param_correct_matched", "q_param_total_matched", "q_param_acc_matched",
        "q_order_correct_matched", "q_order_total_matched", "q_order_acc_matched",
        # LLM-based Sequence
        "q_llm_TP", "q_llm_FP", "q_llm_FN", "q_llm_f1",
        "q_llm_param_correct", "q_llm_param_total", "q_llm_param_acc",
        "q_llm_order_correct", "q_llm_order_total", "q_llm_order_acc",
    ]

    rows = []
    for stem in sorted(per_file_results.keys()):
        result = per_file_results[stem]
        row = {"file": stem}

        if "structured" in result:
            s = result["structured"]
            # LLM results first, fall back to rule-based
            llm_ic = s.get("llm_instrument_completeness")
            llm_ip = s.get("llm_instrument_param_accuracy")
            sc = llm_ic if llm_ic else s["completeness"]
            sp = llm_ip if llm_ip else s["parameter_accuracy"]
            sr = s["reservoir_accuracy"]
            sn = s["natural_accuracy"]

            # Rule-based results (secondary)
            rule_sc = s["completeness"]
            rule_sp = s["parameter_accuracy"]

            s_param_wrong = sp.total - sp.correct
            # TN calculation based on rule-based detail (LLM has no detail)
            wrong_tp_actions = set()
            for d in s["parameter_detail"]:
                if not d["match"]:
                    wrong_tp_actions.add(d["gt_action"])
            s_tn = sc.TP - len(wrong_tp_actions) if not llm_ic else sc.TP
            row.update({
                "s_TP": sc.TP, "s_FP": sc.FP, "s_FN": sc.FN,
                "s_TN": s_tn,
                "s_prec": f"{sc.precision:.4f}",
                "s_rec": f"{sc.recall:.4f}",
                "s_f1": f"{sc.f1:.4f}",
                "s_param_correct": sp.correct,
                "s_param_wrong": s_param_wrong,
                "s_param_total": sp.total,
                "s_param_acc": f"{sp.accuracy:.4f}",
                "s_res_correct": sr.correct, "s_res_total": sr.total,
                "s_res_acc": f"{sr.accuracy:.4f}",
                "s_nat_correct": sn.correct, "s_nat_total": sn.total,
                "s_nat_acc": f"{sn.accuracy:.4f}",
            })

            # Record rule-based results in secondary columns
            row.update({
                "s_llm_TP": rule_sc.TP, "s_llm_FP": rule_sc.FP, "s_llm_FN": rule_sc.FN,
                "s_llm_f1": f"{rule_sc.f1:.4f}",
                "s_llm_param_correct": rule_sp.correct,
                "s_llm_param_total": rule_sp.total,
                "s_llm_param_acc": f"{rule_sp.accuracy:.4f}",
            })

        if "sequence" in result:
            q = result["sequence"]
            # LLM results first, fall back to rule-based
            llm_sc = q.get("llm_completeness")
            llm_sp = q.get("llm_parameter_accuracy")
            llm_so = q.get("llm_execution_order")
            qc = llm_sc if llm_sc else q["completeness"]
            qp = llm_sp if llm_sp else q["parameter_accuracy"]
            qo = llm_so if llm_so else q["execution_order"]

            # Rule-based results (secondary)
            rule_qc = q["completeness"]
            rule_qp = q["parameter_accuracy"]
            rule_qo = q["execution_order"]

            q_param_wrong = qp.total - qp.correct
            wrong_tp_blocks = set()
            for d in q["parameter_detail"]:
                if not d["match"]:
                    wrong_tp_blocks.add(d["gt_block"])
            q_tn = qc.TP - len(wrong_tp_blocks) if not llm_sc else qc.TP
            row.update({
                "q_TP": qc.TP, "q_FP": qc.FP, "q_FN": qc.FN,
                "q_TN": q_tn,
                "q_prec": f"{qc.precision:.4f}",
                "q_rec": f"{qc.recall:.4f}",
                "q_f1": f"{qc.f1:.4f}",
                "q_param_correct": qp.correct,
                "q_param_wrong": q_param_wrong,
                "q_param_total": qp.total,
                "q_param_acc": f"{qp.accuracy:.4f}",
                "q_order_correct": qo.correct_order, "q_order_total": qo.total_blocks,
                "q_order_acc": f"{qo.order_accuracy:.4f}",
                "q_violations": getattr(qo, 'constraint_violations', 0),
            })

            # Matched-only versions
            qp_m = q.get("parameter_accuracy_matched") or qp
            qo_m = q.get("execution_order_matched") or qo
            row.update({
                "q_param_correct_matched": qp_m.correct,
                "q_param_total_matched": qp_m.total,
                "q_param_acc_matched": f"{qp_m.accuracy:.4f}",
                "q_order_correct_matched": qo_m.correct_order,
                "q_order_total_matched": qo_m.total_blocks,
                "q_order_acc_matched": f"{qo_m.order_accuracy:.4f}",
            })

            # Record rule-based results in secondary columns
            row.update({
                "q_llm_TP": rule_qc.TP, "q_llm_FP": rule_qc.FP, "q_llm_FN": rule_qc.FN,
                "q_llm_f1": f"{rule_qc.f1:.4f}",
                "q_llm_param_correct": rule_qp.correct,
                "q_llm_param_total": rule_qp.total,
                "q_llm_param_acc": f"{rule_qp.accuracy:.4f}",
                "q_llm_order_correct": rule_qo.correct_order,
                "q_llm_order_total": rule_qo.total_blocks,
                "q_llm_order_acc": f"{rule_qo.order_accuracy:.4f}",
            })

        rows.append(row)

    path = output_dir / "evaluation_per_file.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Report] {path}")


def _write_errors_only(output_dir, per_file_results):
    """evaluation_errors_only.txt — Only incorrect items collected and printed"""
    lines = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"ERRORS ONLY — {ts}")
    lines.append("=" * 80)

    error_count = 0

    for stem in sorted(per_file_results.keys()):
        result = per_file_results[stem]
        file_errors = []

        if "error" in result:
            file_errors.append(f"  [ERROR] {result['error']}")

        # ── Structured ──
        if "structured" in result:
            s = result["structured"]

            # [5] Completeness — FN (Missing)
            cd = s["completeness_detail"]
            if cd["fn"]:
                file_errors.append("")
                file_errors.append("  [Structured] [5] Missing (FN):")
                for fn in cd["fn"]:
                    file_errors.append(f"    GT #{fn['gt_idx']+1}: {fn['gt_desc']}")
                    file_errors.append(f"      original line: {fn.get('gt_line', '')}")

            # [5] Completeness — FP (Extra)
            if cd["fp"]:
                file_errors.append("")
                file_errors.append("  [Structured] [5] Extra (FP):")
                for fp in cd["fp"]:
                    file_errors.append(f"    Pred #{fp['pred_idx']+1}: {fp['pred_desc']}")
                    file_errors.append(f"      original line: {fp.get('pred_line', '')}")

            # [5] Parameter — WRONG
            wrong_params = [d for d in s["parameter_detail"] if not d["match"]]
            if wrong_params:
                file_errors.append("")
                file_errors.append("  [Structured] [5] Parameter error:")
                for d in wrong_params:
                    file_errors.append(f"    {d['gt_action']} -> {d['param']}: GT={d['gt_val']}  Pred={d['pred_val']}")
                    file_errors.append(f"      GT original:   {d.get('gt_raw_line', '')}")
                    file_errors.append(f"      Pred original: {d.get('pred_raw_line', '')}")

            # Reservoir — WRONG
            wrong_res = [d for d in s["reservoir_detail"] if not d["match"]]
            if wrong_res:
                file_errors.append("")
                file_errors.append("  [Structured] Reservoir error:")
                for d in wrong_res:
                    file_errors.append(f"    R{d['reservoir']}: GT={d['gt_reagent']}  |  Pred={d['pred_reagent']}")

            # Natural [1]~[3] — Missing only (Extra excluded)
            nd = s.get("natural_detail", [])
            missing_items = []
            missing_src_lines = []
            for d in nd:
                if "missing_from_pred" in d:
                    missing_items.extend(d["missing_from_pred"])
                if "missing_source_lines" in d:
                    missing_src_lines.extend(d["missing_source_lines"])
            if missing_items:
                file_errors.append("")
                file_errors.append("  [Structured] [1]~[3] Parameter missing:")
                for item in missing_items:
                    file_errors.append(f"    - {item}")
                if missing_src_lines:
                    file_errors.append("    GT original line:")
                    for src in missing_src_lines:
                        file_errors.append(f"      {src}")

            # LLM-based [5] errors
            llm_id = s.get("llm_instrument_detail", [])
            llm_inst_errors = []
            for d in llm_id:
                for item in d.get("fn_actions", []):
                    llm_inst_errors.append(f"    Missing: {item}")
                for item in d.get("fp_actions", []):
                    llm_inst_errors.append(f"    Extra (FP): {item}")
                for item in d.get("wrong_params", []):
                    llm_inst_errors.append(f"    Parameter: {item}")
            if llm_inst_errors:
                file_errors.append("")
                file_errors.append("  [LLM] [5] Instrument error:")
                file_errors.extend(llm_inst_errors)

        # ── Sequence ──
        if "sequence" in result:
            q = result["sequence"]

            # Completeness — FN
            cd = q["completeness_detail"]
            if cd["fn"]:
                file_errors.append("")
                file_errors.append("  [Sequence] Missing (FN):")
                for fn in cd["fn"]:
                    file_errors.append(f"    GT[{fn['gt_rows']}] {fn['gt_desc']}")

            # Parameter — WRONG
            wrong_params = [d for d in q["parameter_detail"] if not d["match"]]
            if wrong_params:
                file_errors.append("")
                file_errors.append("  [Sequence] Parameter error:")
                for d in wrong_params:
                    gt_rows = d.get('gt_rows', '')
                    pred_rows = d.get('pred_rows', '')
                    file_errors.append(f"    {d['gt_block']} -> {d['param']}: GT={d['gt_val']}  Pred={d['pred_val']}")
                    if gt_rows:
                        file_errors.append(f"      GT rows: {gt_rows}  |  Pred rows: {pred_rows}")

            # Order — violations
            eo = q["execution_order"]
            if eo.constraint_violations > 0:
                file_errors.append("")
                file_errors.append(f"  [Sequence] Order violation: {eo.constraint_violations} case(s)")

            # LLM-based sequence errors
            llm_sd = q.get("llm_detail", [])
            llm_seq_errors = []
            for d in llm_sd:
                for item in d.get("fn_blocks", []):
                    llm_seq_errors.append(f"    Missing: {item}")
                for item in d.get("fp_blocks", []):
                    llm_seq_errors.append(f"    Extra (FP): {item}")
                for item in d.get("wrong_params", []):
                    llm_seq_errors.append(f"    Parameter: {item}")
                for item in d.get("order_errors", []):
                    llm_seq_errors.append(f"    Order: {item}")
            if llm_seq_errors:
                file_errors.append("")
                file_errors.append("  [LLM] Sequence error:")
                file_errors.extend(llm_seq_errors)

        # Print if file has errors
        if file_errors:
            error_count += 1
            lines.append("")
            lines.append("=" * 80)
            lines.append(f"[FILE] {stem}")
            lines.append("=" * 80)
            lines.extend(file_errors)

    # Summary
    lines.insert(1, f"Errors found in {error_count} files")
    lines.insert(2, "=" * 80)

    if error_count == 0:
        lines.append("")
        lines.append("All files OK (No errors)")

    lines.append("")
    lines.append("=" * 80)

    path = output_dir / "evaluation_errors_only.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] {path}")


def _setup_style():
    """Style setup"""
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "sans-serif"]
    for name in candidates:
        found = [f for f in fm.fontManager.ttflist if name in f.name]
        if found:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

# ── Color palette (blue unified) ──
_C = {
    "bg":       "#0b1120",      # background: deep dark navy
    "card":     "#131d36",      # card background
    "card2":    "#1a2744",      # card highlight
    "tp":       "#4da8da",      # TP: sky blue
    "tp_light": "#7ec8e3",      # TP light
    "fn":       "#2b4c7e",      # FN: mid blue (dark)
    "fn_light": "#3a6ba5",
    "fp":       "#3a6ba5",      # FP: mid blue
    "fp_light": "#5a8fc2",
    "tn":       "#1e3250",      # TN: dark blue-gray
    "correct":  "#4da8da",      # Correct: sky blue
    "wrong":    "#2b4c7e",      # Wrong: deep blue
    "text":     "#c8dae8",      # light text
    "text2":    "#7a9bb5",      # secondary text
    "accent":   "#5b9bd5",      # accent: blue
    "white":    "#eaf2f8",      # ivory white
    "grid":     "#1a2744",
}


def _write_confusion_matrix(output_dir, per_file_results):
    """Generate confusion matrix image"""
    _setup_style()

    # ── Aggregation ──
    s_tp = s_fp = s_fn = 0
    s_param_correct = s_param_wrong = 0
    s_res_correct = s_res_wrong = 0
    s_nat_correct = s_nat_wrong = 0
    q_tp = q_fp = q_fn = 0
    q_param_correct = q_param_wrong = 0
    q_order_correct = q_order_wrong = 0

    for result in per_file_results.values():
        if "structured" in result:
            s = result["structured"]
            sc = s["completeness"]
            s_tp += sc.TP; s_fp += sc.FP; s_fn += sc.FN
            sp = s["parameter_accuracy"]
            s_param_correct += sp.correct
            s_param_wrong += (sp.total - sp.correct)
            sr = s["reservoir_accuracy"]
            s_res_correct += sr.correct
            s_res_wrong += (sr.total - sr.correct)
            sn = s["natural_accuracy"]
            s_nat_correct += sn.correct
            s_nat_wrong += (sn.total - sn.correct)
        if "sequence" in result:
            q = result["sequence"]
            qc = q["completeness"]
            q_tp += qc.TP; q_fp += qc.FP; q_fn += qc.FN
            qp = q["parameter_accuracy"]
            q_param_correct += qp.correct
            q_param_wrong += (qp.total - qp.correct)
            qo = q["execution_order"]
            q_order_correct += qo.correct_order
            q_order_wrong += (qo.total_blocks - qo.correct_order)

    model_name = output_dir.name
    fig = plt.figure(figsize=(18, 13), facecolor=_C["bg"])

    # ── Title ──
    fig.text(0.5, 0.97, f"ELISA Protocol Evaluation",
             ha="center", va="top", fontsize=22, fontweight="bold",
             color=_C["white"], fontfamily="sans-serif")
    fig.text(0.5, 0.94, model_name.upper(),
             ha="center", va="top", fontsize=28, fontweight="bold",
             color=_C["accent"], fontfamily="sans-serif")

    # 4 subplots
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.94, top=0.88, bottom=0.05,
                          hspace=0.35, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _draw_cm(ax1, s_tp, s_fp, s_fn, "STRUCTURED  Completeness")
    _draw_bars(ax2,
               [("[5] Param", s_param_correct, s_param_wrong),
                ("Reservoir", s_res_correct, s_res_wrong),
                ("[1]~[3] Param", s_nat_correct, s_nat_wrong)],
               "STRUCTURED  Accuracy")
    _draw_cm(ax3, q_tp, q_fp, q_fn, "SEQUENCE  Completeness")
    _draw_bars(ax4,
               [("Param", q_param_correct, q_param_wrong),
                ("Order", q_order_correct, q_order_wrong)],
               "SEQUENCE  Accuracy")

    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_C["bg"])
    plt.close(fig)
    print(f"[Report] {path}")


def _draw_cm(ax, tp, fp, fn, title):
    """Modern Confusion Matrix"""
    ax.set_facecolor(_C["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Cell data
    cells = [
        (0, 0, tp, "TP", _C["tp"]),
        (0, 1, fn, "FN", _C["fn"]),
        (1, 0, fp, "FP", _C["fp"]),
        (1, 1, "-", "TN", _C["tn"]),
    ]

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.invert_yaxis()

    for row, col, val, label, color in cells:
        # Rounded rectangle
        rect = plt.Rectangle((col - 0.44, row - 0.44), 0.88, 0.88,
                              facecolor=color, alpha=0.25,
                              edgecolor=color, linewidth=2.5,
                              joinstyle="round")
        ax.add_patch(rect)
        # Label
        ax.text(col, row - 0.12, label, ha="center", va="center",
                fontsize=13, color=color, fontweight="bold", alpha=0.7)
        # Value
        val_str = str(val) if val != "-" else "-"
        ax.text(col, row + 0.15, val_str, ha="center", va="center",
                fontsize=26, color=_C["white"], fontweight="bold")

    # Axis labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted\nPositive", "Predicted\nNegative"],
                       fontsize=9, color=_C["text2"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual\nPositive", "Actual\nNegative"],
                       fontsize=9, color=_C["text2"])
    ax.tick_params(length=0)

    # Title
    ax.set_title(title, fontsize=13, fontweight="bold", color=_C["white"],
                 pad=15, loc="left")

    # Metric badges
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    badges = [("P", prec, _C["tp_light"]), ("R", rec, _C["accent"]), ("F1", f1, _C["white"])]
    for i, (lbl, val, clr) in enumerate(badges):
        x = 0.3 + i * 0.5
        ax.text(x, 1.55, f"{lbl} {val:.3f}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=clr,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=clr, alpha=0.15,
                          edgecolor=clr, linewidth=1.2))


def _draw_bars(ax, items, title):
    """Modern horizontal bar chart"""
    ax.set_facecolor(_C["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    names = [it[0] for it in items]
    corrects = [it[1] for it in items]
    wrongs = [it[2] for it in items]
    totals = [c + w for c, w in zip(corrects, wrongs)]
    max_t = max(totals) if totals else 1

    n = len(names)
    y_pos = list(range(n))
    bar_h = 0.55

    for i, (c, w, t) in enumerate(zip(corrects, wrongs, totals)):
        acc = c / t if t > 0 else 1.0

        # Background track
        ax.barh(i, max_t, height=bar_h, color=_C["card"], edgecolor="none",
                zorder=1)
        # Correct bar
        ax.barh(i, c, height=bar_h, color=_C["correct"], edgecolor="none",
                zorder=2, alpha=0.85)
        # Wrong bar
        if w > 0:
            ax.barh(i, w, left=c, height=bar_h, color=_C["wrong"],
                    edgecolor="none", zorder=2, alpha=0.85)

        # correct value
        if c > 0:
            ax.text(c / 2, i, str(c), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=_C["white"], zorder=3)
        # wrong value
        if w > 0:
            w_x = c + w / 2
            ax.text(w_x, i, str(w), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=_C["white"], zorder=3)

        # Accuracy badge (right side)
        badge_color = _C["tp_light"] if acc >= 0.99 else (_C["accent"] if acc >= 0.90 else _C["fn_light"])
        ax.text(max_t * 1.02, i, f"{acc:.4f}",
                ha="left", va="center", fontsize=13, fontweight="bold",
                color=badge_color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12, color=_C["text"], fontweight="bold")
    ax.set_xlim(0, max_t * 1.15)
    ax.tick_params(axis="x", colors=_C["text2"], labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=13, fontweight="bold", color=_C["white"],
                 pad=15, loc="left")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_C["correct"], alpha=0.85, label="Correct"),
        Patch(facecolor=_C["wrong"], alpha=0.85, label="Wrong"),
    ]
    leg = ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
                    facecolor=_C["card"], edgecolor=_C["grid"],
                    labelcolor=_C["text2"])
    leg.get_frame().set_alpha(0.8)
