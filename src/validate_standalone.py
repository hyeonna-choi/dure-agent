#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_standalone.py  —  Standalone validation script (for performance evaluation)

Script for measuring validation accuracy on files with intentionally injected errors.
Runs only the validation stage, independently of the full integrated_main.py pipeline.

Usage:
  # Mode A: original -> structured text validation (completeness)
  python validate_standalone.py --mode structuring \
      --original  path/to/original_protocol.txt \
      --structured path/to/StructuredProtocol.txt \
      --rules     path/to/bioforge_commands_and_rules.xlsx \
      --output    path/to/output_dir

  # Mode B: structured text -> sequence validation (parameter_accuracy + execution_order)
  python validate_standalone.py --mode sequence \
      --structured path/to/StructuredProtocol.txt \
      --sequence   path/to/Seq.xlsx \
      --rules      path/to/bioforge_commands_and_rules.xlsx \
      --output     path/to/output_dir

  # Mode ALL: full validation (completeness + parameter_accuracy + execution_order)
  python validate_standalone.py --mode all \
      --original   path/to/original_protocol.txt \
      --structured path/to/StructuredProtocol.txt \
      --sequence   path/to/Seq.xlsx \
      --rules      path/to/bioforge_commands_and_rules.xlsx \
      --output     path/to/output_dir

  # Options
  --model      Claude model name (default: claude-3-5-sonnet-20241022)
  --max-tokens Maximum tokens (default: 2048)
  --label      Label to attach to the result files (e.g., "error_test_01")
"""

import os
import sys
import json
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

# Add the project root (this directory) to the import path.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bioforge.core.validator_claude import ClaudeValidator


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def _read_text_file(path: Path) -> str:
    """Read a text file and return its contents."""
    return path.read_text(encoding="utf-8", errors="ignore")


def _format_stage_report(stage_name: str, stage_data: dict) -> list:
    """Convert a single stage result into report lines."""
    lines = []
    result = stage_data.get("result", "UNKNOWN")
    issues = stage_data.get("issues", []) or []
    error = stage_data.get("_error", "")

    lines.append(f"  [{stage_name}] => {result}")
    if error:
        lines.append(f"    _error: {error}")
    lines.append(f"    issues_count: {len(issues)}")

    for i, issue in enumerate(issues, start=1):
        if isinstance(issue, dict):
            exp = issue.get("expected", "")
            obs = issue.get("observed", "")
            idx = issue.get("indices", [])
            lines.append(f"    {i}) expected: {exp}")
            lines.append(f"       observed: {obs}")
            if idx:
                lines.append(f"       indices: {idx}")
        else:
            lines.append(f"    {i}) {str(issue)}")

    return lines


def _write_report(output_dir: Path, label: str, mode: str,
                  stages_run: list, report_dict: dict,
                  original_path: str = "", structured_path: str = "",
                  sequence_path: str = "", rules_path: str = ""):
    """Save the validation result report to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation_{mode}_{label}_{ts}.txt"
    report_path = output_dir / filename

    lines = []
    lines.append("=" * 80)
    lines.append(f"[STANDALONE VALIDATION REPORT]")
    lines.append(f"  timestamp : {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  mode      : {mode}")
    lines.append(f"  label     : {label}")
    lines.append(f"  stages    : {stages_run}")
    lines.append(f"  overall   : {report_dict.get('overall', 'N/A')}")
    lines.append("")
    lines.append(f"  original  : {original_path}")
    lines.append(f"  structured: {structured_path}")
    lines.append(f"  sequence  : {sequence_path}")
    lines.append(f"  rules     : {rules_path}")
    lines.append("=" * 80)
    lines.append("")

    # Per-stage results
    stages = report_dict.get("stages", {}) or {}
    for stage_name in stages_run:
        s = stages.get(stage_name, {})
        lines.extend(_format_stage_report(stage_name, s))
        lines.append("")

    # Must Fix
    must_fix = report_dict.get("must_fix", []) or []
    lines.append("[Must Fix]")
    if not must_fix:
        lines.append("  (none)")
    else:
        for i, item in enumerate(must_fix, start=1):
            lines.append(f"  {i}) {json.dumps(item, ensure_ascii=False)}")
    lines.append("")

    # Feedback
    fb = report_dict.get("feedback_to_regenerator", {}) or {}
    lines.append("[Feedback To Regenerator]")
    lines.append(f"  target: {fb.get('target', '')}")
    instr = (fb.get("instructions", "") or "").strip()
    lines.append(f"  instructions: {instr if instr else '(empty)'}")
    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Also save the JSON result separately (for programmatic analysis)
    json_path = output_dir / f"validation_{mode}_{label}_{ts}.json"
    json_output = {
        "timestamp": dt.datetime.now().isoformat(),
        "mode": mode,
        "label": label,
        "stages_run": stages_run,
        "inputs": {
            "original": original_path,
            "structured": structured_path,
            "sequence": sequence_path,
            "rules": rules_path,
        },
        "result": report_dict,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    return report_path, json_path


# ─────────────────────────────────────────────────────────────────────
# Validation execution per mode
# ─────────────────────────────────────────────────────────────────────

def run_structuring_validation(validator: ClaudeValidator,
                                original_text: str,
                                structured_text: str,
                                max_tokens: int = 2048,
                                temperature: float = 0.0) -> dict:
    """
    Mode A: original -> structured text validation
    Runs only the completeness stage (without a sequence).
    Checks whether all content of the original protocol is reflected in the structured text.
    """
    print("[Mode A] original -> structured text validation (completeness)")
    print("-" * 50)

    # Call completeness only (seq_dump is an empty string — not needed for structured text validation)
    # The stage task is redefined to be structuring-specific
    stage_name = "completeness"
    result = validator._call_stage(
        stage_name=stage_name,
        original_text=original_text,
        structured_text=structured_text,
        seq_dump="(sequence not provided — structuring validation only)",
        attempt=1,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    stages = {
        "completeness": {
            "result": result.get("result", "UNKNOWN"),
            "issues": result.get("issues", []),
        }
    }
    if result.get("_error"):
        stages["completeness"]["_error"] = result["_error"]

    overall = result.get("result", "UNKNOWN")
    if overall not in ("PASS", "FAIL"):
        overall = "FAIL"

    all_issues = result.get("issues", [])
    must_fix = all_issues[:3] if overall == "FAIL" else []

    report = {
        "overall": overall,
        "stages": stages,
        "must_fix": must_fix,
        "feedback_to_regenerator": {
            "target": "STRUCTURING" if overall == "FAIL" else "",
            "instructions": "\n".join(
                f"- Expected: {iss.get('expected','')} / Observed: {iss.get('observed','')}"
                for iss in must_fix if isinstance(iss, dict)
            ) if must_fix else "",
        },
    }

    _print_result(overall, stages, ["completeness"])
    return report


def run_sequence_validation(validator: ClaudeValidator,
                             structured_text: str,
                             seq_df: pd.DataFrame,
                             original_text: str = "",
                             max_tokens: int = 2048,
                             temperature: float = 0.0) -> dict:
    """
    Mode B: structured text -> sequence validation
    Runs the parameter_accuracy + execution_order stages.
    Checks whether the structured protocol was correctly converted into sequence commands.
    """
    print("[Mode B] structured text -> sequence validation (parameter_accuracy + execution_order)")
    print("-" * 50)

    from bioforge.core.validator_claude import _sequence_df_to_dump
    seq_dump = _sequence_df_to_dump(seq_df)

    # If original_text is not provided, use structured_text as a fallback
    if not original_text:
        original_text = "(original not provided — using structured text as reference)"

    stages = {}
    all_issues = []

    for stage_name in ["parameter_accuracy", "execution_order"]:
        result = validator._call_stage(
            stage_name=stage_name,
            original_text=original_text,
            structured_text=structured_text,
            seq_dump=seq_dump,
            attempt=1,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        stages[stage_name] = {
            "result": result.get("result", "UNKNOWN"),
            "issues": result.get("issues", []),
        }
        if result.get("_error"):
            stages[stage_name]["_error"] = result["_error"]
        all_issues.extend(result.get("issues", []))

    # Determine overall result
    stage_results = [s["result"] for s in stages.values()]
    if "FAIL" in stage_results:
        overall = "FAIL"
    elif "UNKNOWN" in stage_results:
        overall = "FAIL"
    else:
        overall = "PASS"

    must_fix = all_issues[:3] if overall == "FAIL" else []

    if overall == "FAIL":
        fail_stages = [name for name, s in stages.items() if s["result"] == "FAIL"]
        target = "MAPPING" if "parameter_accuracy" in fail_stages else "BOTH"
    else:
        target = ""

    report = {
        "overall": overall,
        "stages": stages,
        "must_fix": must_fix,
        "feedback_to_regenerator": {
            "target": target,
            "instructions": "\n".join(
                f"- Expected: {iss.get('expected','')} / Observed: {iss.get('observed','')}"
                for iss in must_fix if isinstance(iss, dict)
            ) if must_fix else "",
        },
    }

    _print_result(overall, stages, ["parameter_accuracy", "execution_order"])
    return report


def run_all_validation(validator: ClaudeValidator,
                        original_text: str,
                        structured_text: str,
                        seq_df: pd.DataFrame,
                        max_tokens: int = 2048,
                        temperature: float = 0.0) -> dict:
    """
    Mode ALL: full validation (identical to the existing validate())
    completeness + parameter_accuracy + execution_order
    """
    print("[Mode ALL] full validation (completeness + parameter_accuracy + execution_order)")
    print("-" * 50)

    report = validator.validate(
        original_text=original_text,
        structured_text=structured_text,
        seq_df=seq_df,
        attempt=1,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    stages = report.get("stages", {})
    overall = report.get("overall", "UNKNOWN")
    _print_result(overall, stages, ["completeness", "parameter_accuracy", "execution_order"])
    return report


def _print_result(overall: str, stages: dict, stage_names: list):
    """Print a result summary to the console."""
    print("")
    print("=" * 50)
    print(f"  OVERALL: {overall}")
    print("=" * 50)
    for name in stage_names:
        s = stages.get(name, {})
        result = s.get("result", "UNKNOWN")
        issues = s.get("issues", []) or []
        mark = "[OK]" if result == "PASS" else "[FAIL]" if result == "FAIL" else "[WARN]"
        print(f"  {mark} {name}: {result} ({len(issues)} issues)")
        for i, iss in enumerate(issues, 1):
            if isinstance(iss, dict):
                print(f"      {i}) expected: {iss.get('expected','')}")
                print(f"         observed: {iss.get('observed','')}")
            else:
                print(f"      {i}) {iss}")
    print("=" * 50)
    print("")


# ─────────────────────────────────────────────────────────────────────
# Batch execution (multiple files at once)
# ─────────────────────────────────────────────────────────────────────

def run_batch(args):
    """
    Automatically validate all files in a folder via the --batch-dir option.
    Example folder structure:
      batch_dir/
        test_01/
          original.txt
          structured.txt
          sequence.xlsx
        test_02/
          original.txt
          structured.txt
          sequence.xlsx
    """
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        print(f"[ERROR] Batch folder does not exist: {batch_dir}")
        sys.exit(1)

    rules_path = Path(args.rules)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = ClaudeValidator(
        model=args.model,
        rules_xlsx_path=str(rules_path),
    )
    print(f"[INFO] Model: {validator.model}")
    print(f"[INFO] Batch folder: {batch_dir}")
    print("")

    test_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
    if not test_dirs:
        print("[WARN] No test subfolders found.")
        return

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"[BATCH VALIDATION SUMMARY]  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"  mode: {args.mode}  |  model: {validator.model}")
    summary_lines.append(f"  batch_dir: {batch_dir}")
    summary_lines.append(f"  total tests: {len(test_dirs)}")
    summary_lines.append("=" * 80)
    summary_lines.append("")

    results = []

    for test_dir in test_dirs:
        label = test_dir.name
        print(f"\n{'─'*60}")
        print(f"[TEST] {label}")
        print(f"{'─'*60}")

        # Locate files
        orig_file = _find_file(test_dir, ["original.txt", "original_protocol.txt", "*original*"])
        struct_file = _find_file(test_dir, ["structured.txt", "*StructuredProtocol*", "*structured*"])
        seq_file = _find_file(test_dir, ["sequence.xlsx", "*Seq.xlsx", "*sequence*"])

        original_text = _read_text_file(orig_file) if orig_file else ""
        structured_text = _read_text_file(struct_file) if struct_file else ""
        seq_df = pd.read_excel(seq_file) if seq_file else None

        mode = args.mode
        try:
            if mode == "structuring":
                if not structured_text:
                    print(f"  [SKIP] structured file missing")
                    continue
                report = run_structuring_validation(
                    validator, original_text, structured_text,
                    args.max_tokens, 0.0
                )
                stages_run = ["completeness"]
            elif mode == "sequence":
                if not structured_text or seq_df is None:
                    print(f"  [SKIP] structured or sequence file missing")
                    continue
                report = run_sequence_validation(
                    validator, structured_text, seq_df, original_text,
                    args.max_tokens, 0.0
                )
                stages_run = ["parameter_accuracy", "execution_order"]
            else:  # all
                if not structured_text or seq_df is None:
                    print(f"  [SKIP] required files missing")
                    continue
                report = run_all_validation(
                    validator, original_text, structured_text, seq_df,
                    args.max_tokens, 0.0
                )
                stages_run = ["completeness", "parameter_accuracy", "execution_order"]

            # Save individual result
            test_output = output_dir / label
            rpt_path, json_path = _write_report(
                test_output, label, mode, stages_run, report,
                str(orig_file or ""), str(struct_file or ""),
                str(seq_file or ""), str(rules_path),
            )
            print(f"  -> report: {rpt_path}")

            overall = report.get("overall", "UNKNOWN")
            results.append({"label": label, "overall": overall, "report": report})

            # Add to summary
            summary_lines.append(f"  [{overall}] {label}")
            stages = report.get("stages", {})
            for sn in stages_run:
                s = stages.get(sn, {})
                sr = s.get("result", "UNKNOWN")
                ic = len(s.get("issues", []) or [])
                summary_lines.append(f"        {sn}: {sr} ({ic} issues)")

        except Exception as e:
            print(f"  [ERROR] {e}")
            summary_lines.append(f"  [ERROR] {label}: {e}")
            results.append({"label": label, "overall": "ERROR", "error": str(e)})

    # Batch summary
    summary_lines.append("")
    summary_lines.append("-" * 40)
    total = len(results)
    passed = sum(1 for r in results if r.get("overall") == "PASS")
    failed = sum(1 for r in results if r.get("overall") == "FAIL")
    errors = sum(1 for r in results if r.get("overall") == "ERROR")
    summary_lines.append(f"  TOTAL: {total}  |  PASS: {passed}  |  FAIL: {failed}  |  ERROR: {errors}")
    if total > 0:
        summary_lines.append(f"  Accuracy: {passed}/{total} = {passed/total*100:.1f}%")
    summary_lines.append("=" * 80)

    summary_path = output_dir / f"batch_summary_{args.mode}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n{'='*60}")
    print(f"[BATCH COMPLETE] {total} tests | PASS: {passed} | FAIL: {failed} | ERROR: {errors}")
    if total > 0:
        print(f"  Accuracy: {passed}/{total} = {passed/total*100:.1f}%")
    print(f"  Summary report: {summary_path}")
    print(f"{'='*60}")


def _find_file(directory: Path, patterns: list):
    """Find and return a file in the folder matching the patterns."""
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


# ─────────────────────────────────────────────────────────────────────
# CLI main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone validation script — for performance evaluation (runs only the validation stage)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode", required=True,
        choices=["structuring", "sequence", "all"],
        help=(
            "Validation mode:\n"
            "  structuring : original -> structured text validation (completeness)\n"
            "  sequence    : structured text -> sequence validation (parameter_accuracy + execution_order)\n"
            "  all         : full validation (all three stages)"
        ),
    )

    # Input files
    parser.add_argument("--original", type=str, default="",
                        help="Path to the original protocol text file")
    parser.add_argument("--structured", type=str, default="",
                        help="Path to the structured protocol text file")
    parser.add_argument("--sequence", type=str, default="",
                        help="Path to the sequence Excel file (.xlsx)")
    parser.add_argument("--rules", type=str,
                        default=str(PROJECT_ROOT / "data" / "rules" / "bioforge_commands_and_rules.xlsx"),
                        help="Path to the rules Excel file")

    # Output
    parser.add_argument("--output", type=str, default="./validation_results",
                        help="Result output folder")
    parser.add_argument("--label", type=str, default="test",
                        help="Label to attach to the result files (e.g., error_test_01)")

    # Model settings
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022",
                        help="Claude model name")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum number of tokens")

    # Batch mode
    parser.add_argument("--batch-dir", type=str, default="",
                        help="Folder path for batch validation (automatic validation per subfolder)")

    args = parser.parse_args()

    # ── Batch mode ──
    if args.batch_dir:
        run_batch(args)
        return

    # ── Single-run mode ──
    rules_path = Path(args.rules)
    if not rules_path.exists():
        print(f"[ERROR] Rules file not found: {rules_path}")
        sys.exit(1)

    validator = ClaudeValidator(
        model=args.model,
        rules_xlsx_path=str(rules_path),
    )
    print(f"[INFO] Model: {validator.model}")

    output_dir = Path(args.output)
    mode = args.mode

    # Load input files
    original_text = ""
    structured_text = ""
    seq_df = None

    if args.original:
        orig_path = Path(args.original)
        if orig_path.exists():
            original_text = _read_text_file(orig_path)
            print(f"[INFO] Loaded original: {orig_path} ({len(original_text)} chars)")
        else:
            print(f"[WARN] Original file not found: {orig_path}")

    if args.structured:
        struct_path = Path(args.structured)
        if struct_path.exists():
            structured_text = _read_text_file(struct_path)
            print(f"[INFO] Loaded structured: {struct_path} ({len(structured_text)} chars)")
        else:
            print(f"[ERROR] Structured file not found: {struct_path}")
            sys.exit(1)

    if args.sequence:
        seq_path = Path(args.sequence)
        if seq_path.exists():
            seq_df = pd.read_excel(seq_path)
            print(f"[INFO] Loaded sequence: {seq_path} ({len(seq_df)} rows)")
        else:
            print(f"[WARN] Sequence file not found: {seq_path}")

    print("")

    # ── Execution per mode ──
    if mode == "structuring":
        if not structured_text:
            print("[ERROR] The --structured path is required.")
            sys.exit(1)
        report = run_structuring_validation(
            validator, original_text, structured_text,
            args.max_tokens, 0.0
        )
        stages_run = ["completeness"]

    elif mode == "sequence":
        if not structured_text:
            print("[ERROR] The --structured path is required.")
            sys.exit(1)
        if seq_df is None:
            print("[ERROR] The --sequence path is required.")
            sys.exit(1)
        report = run_sequence_validation(
            validator, structured_text, seq_df, original_text,
            args.max_tokens, 0.0
        )
        stages_run = ["parameter_accuracy", "execution_order"]

    elif mode == "all":
        if not structured_text:
            print("[ERROR] The --structured path is required.")
            sys.exit(1)
        if seq_df is None:
            print("[ERROR] The --sequence path is required.")
            sys.exit(1)
        report = run_all_validation(
            validator, original_text, structured_text, seq_df,
            args.max_tokens, 0.0
        )
        stages_run = ["completeness", "parameter_accuracy", "execution_order"]

    # ── Save results ──
    rpt_path, json_path = _write_report(
        output_dir, args.label, mode, stages_run, report,
        args.original, args.structured, args.sequence, args.rules,
    )

    print(f"[SAVED] Text report: {rpt_path}")
    print(f"[SAVED] JSON report : {json_path}")


if __name__ == "__main__":
    main()
