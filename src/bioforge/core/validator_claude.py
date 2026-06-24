import os
import re
import json
import time
import pandas as pd
from typing import Dict, Any, List, Optional
from anthropic import Anthropic

# Llama -> Together AI base URL
_LLAMA_MODEL_MAP = {
    "llama-3.3-70b":    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "llama-4-scout":    "kimmairobot_46cf/meta-llama/Llama-4-Scout-17B-16E-a1dbbc6e",
}
_TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def _fix_broken_json(raw: str) -> str:
    """
    Attempt to repair broken JSON produced by Claude.
    """
    # Remove trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    # Close truncated JSON
    opens = fixed.count("{") + fixed.count("[")
    closes = fixed.count("}") + fixed.count("]")
    if opens > closes:
        diff = opens - closes
        for _ in range(diff):
            if fixed.rstrip().endswith(","):
                fixed = fixed.rstrip().rstrip(",")
            last_open = max(fixed.rfind("{"), fixed.rfind("["))
            last_close = max(fixed.rfind("}"), fixed.rfind("]"))
            if last_open > last_close:
                closer = "}" if fixed[last_open] == "{" else "]"
            else:
                closer = "}"
            fixed += closer
    return fixed


def _extract_json(text: str) -> dict:
    """
    Extract and parse only the JSON portion from a Claude response.
    """
    # 1) ```json ... ``` code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.S)
    raw = m.group(1).strip() if m else None

    # 2) First { to last }
    if not raw:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = text[start:end + 1]
        else:
            raw = text

    # 3) Parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 4) Retry after repair
    fixed = _fix_broken_json(raw)
    return json.loads(fixed)


def _check_preparation_boundary(structured_text: str) -> list:
    """
    Detect section-classification errors in STRUCTURED_OUTPUT at the code level.
    Three checks:
      1) Whether [5] contains coating/blocking keywords (overinclusion)
      2) Whether [4] Reservoir contains coating/blocking reagents
      3) Whether the first execution line of [5] starts with coating/blocking
    """
    extra_issues = []

    # Shared keyword definitions
    COATING_KW = [
        "capture ab", "capture antibody", "코팅", "coating",
        "capture 항체", "포획 항체", "포획항체",
    ]
    BLOCKING_KW = [
        "block buffer", "봉쇄", "블로킹",
        "block 버퍼", "블록 버퍼", "블록버퍼",
    ]
    # For validating the first line of [5] -- start patterns of coating/blocking
    PREP_START_PATTERNS = [
        r"capture\s*(ab|antibody|항체)",
        r"block\s*buffer",
        r"blocking",
        r"봉쇄",
        r"코팅",
    ]

    # -- Section extraction --
    section5 = ""
    instrument_match = re.search(r"<INSTRUMENT>(.*?)</INSTRUMENT>", structured_text, re.S)
    if instrument_match:
        instrument_text = instrument_match.group(1)
        section5_match = re.search(r"\[5\](.*?)(?:\[6\]|$)", instrument_text, re.S)
        if section5_match:
            section5 = section5_match.group(1)

    section4 = ""
    section2 = ""
    natural_match = re.search(r"<NATURAL>(.*?)</NATURAL>", structured_text, re.S)
    if natural_match:
        natural_text = natural_match.group(1)
        s4 = re.search(r"\[4\](.*?)(?:\[5\]|<|$)", natural_text, re.S)
        if s4:
            section4 = s4.group(1)
        s2 = re.search(r"\[2\](.*?)(?:\[3\]|$)", natural_text, re.S)
        if s2:
            section2 = s2.group(1)

    section5_lower = section5.lower()
    section4_lower = section4.lower()
    section2_lower = section2.lower()

    # -- Check 1: detect coating/blocking keywords in [5] --
    for kw in COATING_KW:
        if kw in section5_lower:
            extra_issues.append({
                "expected": "Coating (Capture Ab) must only appear in the [2] user manual steps. Must be removed from [5].",
                "observed": f"Coating-related keyword '{kw}' found in the [5] automation block",
                "indices": [],
            })
            break

    for kw in BLOCKING_KW:
        if kw in section5_lower:
            extra_issues.append({
                "expected": "Blocking (Block Buffer) must only appear in the [2] user manual steps. Must be removed from [5].",
                "observed": f"Blocking-related keyword '{kw}' found in the [5] automation block",
                "indices": [],
            })
            break

    # -- Check 2: detect coating/blocking reagents in the [4] Reservoir mapping --
    reservoir_coating = ["capture ab", "capture antibody", "코팅", "포획 항체", "포획항체"]
    reservoir_blocking = ["block buffer", "blocking buffer", "블록 버퍼", "블록버퍼", "봉쇄"]

    for kw in reservoir_coating:
        if kw in section4_lower:
            extra_issues.append({
                "expected": "Capture Ab (for coating) must not be included in the [4] Reservoir mapping. The user performs it manually.",
                "observed": f"Coating reagent '{kw}' found in the [4] Reservoir mapping",
                "indices": [],
            })
            break

    for kw in reservoir_blocking:
        if kw in section4_lower:
            extra_issues.append({
                "expected": "Block Buffer (for blocking) must not be included in the [4] Reservoir mapping. The user performs it manually.",
                "observed": f"Blocking reagent '{kw}' found in the [4] Reservoir mapping",
                "indices": [],
            })
            break

    # -- Check 3: validate the first execution line of [5] (two stages) --
    if section5.strip():
        lines_5 = [ln.strip() for ln in section5.strip().split("\n") if ln.strip() and ln.strip().startswith("-")]
        if lines_5:
            first_line = lines_5[0].lower()

            # 3a: immediate FAIL if it starts with a coating/blocking keyword
            for pat in PREP_START_PATTERNS:
                if re.search(pat, first_line):
                    extra_issues.append({
                        "expected": "The first step of [5] must be a dispense after coating/blocking (Standard/Sample/Assay Diluent).",
                        "observed": f"First line of [5] starts with coating/blocking: '{lines_5[0][:80]}'",
                        "indices": [],
                    })
                    break

            # 3b: if the first line of [5] does not start with "dispense" but with
            #     "solution removal" or "wash" -> the last step of blocking
            #     (solution removal + wash) has spilled over into [5].
            #     A valid [5] must always start with a Reservoir dispense.
            is_dispense_start = bool(re.search(
                r"reservoir|분주|assay diluent|배양",
                first_line
            ))
            is_removal_or_wash = bool(re.search(
                r"용액\s*제거|세척|wash|remove|aspirat",
                first_line
            ))
            if not is_dispense_start and is_removal_or_wash:
                extra_issues.append({
                    "expected": "The first step of [5] must be a Reservoir dispense (Standard/Sample/Assay Diluent). "
                               "Solution removal/wash is the last step of [2] coating/blocking, so it must not be included in [5].",
                    "observed": f"[5] starts with solution removal/wash instead of a dispense: '{lines_5[0][:80]}'",
                    "indices": [],
                })

    # -- Check 4: if [2] has no coating but [5] has an overnight incubation -> coating may have gone into [5] --
    has_coating_in_2 = any(kw in section2_lower for kw in ["capture ab", "capture antibody", "코팅", "포획 항체"])
    has_overnight_in_5 = bool(re.search(r"overnight|오버나이트|밤새", section5_lower))
    if not has_coating_in_2 and has_overnight_in_5:
        extra_issues.append({
            "expected": "Coating (including overnight incubation) must be in [2]. [2] has no coating while [5] has an overnight incubation.",
            "observed": "No coating-related content in [2] + overnight incubation found in [5] -> judged that coating was incorrectly placed in [5]",
            "indices": [],
        })

    # -- Check 5: if there is solution removal/wash after Stop Solution -> GPT hallucination --
    if section5.strip():
        lines_5 = [ln.strip() for ln in section5.strip().split("\n") if ln.strip()]
        stop_found = False
        for i, line in enumerate(lines_5):
            line_lower = line.lower()
            # Detect the Stop Solution line
            if any(kw in line_lower for kw in ["stop solution", "stop sol", "1m hcl", "1 m hcl", "2n h₂so₄", "2n h2so4", "h2so4"]):
                stop_found = True
                continue
            # Catch solution removal/wash after Stop
            if stop_found:
                if any(kw in line_lower for kw in ["용액 제거", "세척", "wash", "aspirat"]):
                    extra_issues.append({
                        "expected": "After dispensing Stop Solution, perform reading only. No solution removal/wash (it destroys the experiment results).",
                        "observed": f"Unnecessary step found after Stop Solution: '{line[:80]}'",
                        "indices": [],
                    })
                    break  # Report only one

    return extra_issues


def _filter_sequence_false_positives(issues: list, seq_df: pd.DataFrame) -> list:
    """
    Remove Claude hallucination false positives by directly inspecting the sequence at the code level.
    Example: an 'incubation missing' issue when an incubation block actually exists in the sequence.
    """
    if seq_df is None or seq_df.empty:
        return issues

    # Build a command list (for fast lookup)
    commands = []
    for _, row in seq_df.iterrows():
        cmd = str(row.get("Command", "")).strip()
        commands.append(cmd)
    total = len(commands)

    filtered = []
    for issue in issues:
        if not isinstance(issue, dict):
            filtered.append(issue)
            continue

        observed = (issue.get("observed", "") or "").lower()
        expected = (issue.get("expected", "") or "").lower()
        combined = observed + " " + expected

        # -- (1) Incubation missing false-positive check --
        is_incubation_missing_claim = (
            ("incubation" in combined or "wait_motion" in combined or "배양" in combined)
            and ("missing" in combined or "not found" in combined
                 or "no incubation" in combined or "no wait_motion" in combined
                 or "without" in combined or "없" in combined)
        )

        if is_incubation_missing_claim:
            indices = issue.get("indices", [])
            if indices:
                # Search for the incubation block pattern within +/-10 of the indices mentioned in the issue
                search_start = max(0, min(indices) - 2)
                search_end = min(total - 3, max(indices) + 10)
                found_incubation = False
                for i in range(search_start, search_end):
                    if (i + 3 < total
                        and "TRANS" in commands[i]
                        and "TEMPPLATE_ON" in commands[i + 1]
                        and "WAIT_MOTION" in commands[i + 2]
                        and "TRANS" in commands[i + 3]):
                        found_incubation = True
                        break
                if found_incubation:
                    # An incubation block actually exists -> remove false positive
                    continue

        # -- (2) Reservoir swap false-positive filter --
        # A Reservoir-number swap between common reagents is not an error
        is_reservoir_swap_issue = (
            ("reservoir" in combined or "reservoir" in combined.lower())
            and ("swap" in combined or "번호" in combined or "순서" in combined
                 or "column" in combined or "position" in combined
                 or "wash buffer" in combined or "assay diluent" in combined)
            and ("wrong" in combined or "incorrect" in combined or "mismatch" in combined
                 or "different" in combined or "다르" in combined or "불일치" in combined)
        )
        if is_reservoir_swap_issue:
            continue

        # -- (3) Shaking parameter false-positive filter --
        # rpm/time differences in shaking/tap are not errors (default 200 rpm / 10 s applied automatically)
        is_shaking_param_issue = (
            ("shaking" in combined or "shake" in combined or "tap" in combined or "혼합" in combined)
            and ("rpm" in combined or "초" in combined or "second" in combined
                 or "time" in combined or "시간" in combined or "duration" in combined)
        )
        if is_shaking_param_issue:
            continue

        # -- (4) User-defined Calibrant false-positive filter --
        # When plate_mapper allocates more Calibrant Reservoirs than the PDF,
        # GPT labeling them as "사용자 지정 Calibrant (User-defined)" is correct behavior.
        # The validator flagging it as a "calibrant not in PDF" is a false positive.
        # Note: volume/concentration errors are NOT filtered (they may be real errors).
        is_user_defined_calibrant_issue = (
            ("user-defined" in combined or "user defined" in combined
             or "사용자 지정" in combined)
            and ("calibrant" in combined or "standard" in combined)
        )
        # Exclude volume/concentration-related issues from the filter (preserve real errors)
        is_volume_or_concentration_issue = (
            "volume" in combined or "µl" in combined or "ul" in combined
            or "concentration" in combined or "농도" in combined or "볼륨" in combined
        )
        if is_user_defined_calibrant_issue and not is_volume_or_concentration_issue:
            # If only existence is in question -> remove false positive
            continue

        filtered.append(issue)

    return filtered


def _code_level_sequence_checks(structured_text: str, seq_df: pd.DataFrame) -> list:
    """
    Code-level sequence validation: detect issues by analyzing the sequence directly, without relying on Claude.
    - Wash count validation
    - Volume validation (structured vs sequence)
    - Detection of wash/removal after Stop Solution
    - Detection of an unfilled <INSTRUMENT> section
    - Minimum sequence row-count validation
    """
    import re as _cl_re
    issues = []

    # -- (0) Detect an unfilled <INSTRUMENT> section --
    # When GPT fails to generate the [5] automation section and fills it only with '미기재' (unspecified)
    inst_match0 = _cl_re.search(r'<INSTRUMENT>(.*?)</INSTRUMENT>', structured_text, flags=_cl_re.S)
    if inst_match0:
        inst_body = inst_match0.group(1).strip()
        # Strip whitespace and zero-width spaces, then check the actual content
        inst_clean = _cl_re.sub(r'[\s\u200b\u00a0]+', '', inst_body)
        if inst_clean in ('', '미기재', 'N/A', 'NA'):
            issues.append({
                "expected": (
                    "The [5] automation sequence (<INSTRUMENT> section) must contain actual step-by-step dispense/wash/incubation instructions"
                ),
                "observed": (
                    f"The <INSTRUMENT> section is filled only with '{inst_body[:30]}' -- "
                    "suspected that GPT failed to generate the [5] automation content. "
                    "sequence_builder only outputs the epilogue (plate transfer)."
                ),
                "indices": [],
            })
            # If [5] is absent, the remaining sequence checks are meaningless, so return early
            return issues

    if seq_df is None or seq_df.empty:
        return issues

    commands = []
    params_list = []
    for _, row in seq_df.iterrows():
        cmd = str(row.get("Command", "")).strip()
        prm = str(row.get("Input Parameters", "")) if not pd.isna(row.get("Input Parameters", "")) else ""
        commands.append(cmd)
        params_list.append(prm.strip())
    total = len(commands)

    # -- (1) Wash count validation --
    # Extract the "세척 N회" (wash N times) pattern from structured
    wash_matches = _cl_re.findall(r'세척\s*(\d+)\s*회', structured_text)
    if wash_matches:
        expected_wash_count = int(wash_matches[0])  # First wash count (usually all identical)

        # Count WASH_BLOCK patterns in the sequence
        # Wash block = a group of consecutive wash cycles
        # Dispense/incubation/removal etc. are interleaved between wash blocks
        # We need to count the "number of rounds" within each wash block
        # Method: count via ASPIRATE_FROM_PLATE (slot 5, wash reservoir col), etc.
        # Simpler approach: match each "wash N times" line in structured with wash blocks in the sequence

        # Simplification: identify independent wash block groups across the whole sequence
        # A wash block is a consecutive SELECT_CHANNEL / INSTALL / ASPIRATE(slot 5, wash res) / DISPENSE / ASPIRATE(plate) / DISPENSE(waste) / EJECT pattern
        # Here we keep it simple: rather than counting consecutive wash repeats from the same reservoir col,
        # we validate based only on the structured text, so skip (supplements Claude)
        pass  # Wash count is structurally complex, so keep handling it at the prompt level

    # -- (2) Common reagent volume validation --
    # Extract the "Reservoir N (reagent) X µL/well dispense" pattern from structured [5]
    inst_match = _cl_re.search(r'<INSTRUMENT>(.*?)</INSTRUMENT>', structured_text, flags=_cl_re.S)
    if inst_match:
        inst_text = inst_match.group(1)
        sec5_match = _cl_re.search(r'\[5\](.*?)(?:\[6\]|$)', inst_text, flags=_cl_re.S)
        if sec5_match:
            sec5 = sec5_match.group(1)
            # Extract Reservoir number + volume from common reagent dispense lines (excluding type-based "지정된 웰")
            dispense_lines = _cl_re.findall(
                r'Reservoir\s+(\d+)\s*\([^)]*\)\s*(?:전체 웰에\s*)?(\d+)\s*µL/웰\s*분주',
                sec5
            )
            for res_num_str, vol_str in dispense_lines:
                res_num = int(res_num_str)
                expected_vol = int(vol_str)

                # Check the DISPENSE volume from this Reservoir (slot 5, col=res_num) in the sequence
                for i, cmd in enumerate(commands):
                    if "DISPENSE_INTO_PLATE" in cmd:
                        prm = params_list[i].split()
                        if len(prm) >= 4:
                            try:
                                slot = int(prm[0])
                                vol_actual = int(prm[3])
                                # Check whether the ASPIRATE immediately before this DISPENSE is slot 5, col=res_num
                                # (trace back to confirm it came from this reservoir)
                                for j in range(i - 1, max(i - 10, -1), -1):
                                    if "ASPIRATE_FROM_PLATE" in commands[j]:
                                        asp_prm = params_list[j].split()
                                        if len(asp_prm) >= 2:
                                            asp_slot = int(asp_prm[0])
                                            asp_col = int(asp_prm[1])
                                            if asp_slot == 5 and asp_col == res_num:
                                                if vol_actual != expected_vol:
                                                    issues.append({
                                                        "expected": f"Reservoir {res_num} dispense volume must be {expected_vol} µL (per structured [5])",
                                                        "observed": f"DISPENSE volume={vol_actual} µL at sequence index {i} (from Reservoir {res_num})",
                                                        "indices": [i],
                                                    })
                                        break
                            except (ValueError, IndexError):
                                pass
                        break  # Check only the first DISPENSE (volumes must be identical)

    # -- (3) Sequence-level validation of wash/removal after Stop Solution --
    # Extract the Stop Solution Reservoir number from structured [4]
    nat_match = _cl_re.search(r'<NATURAL>(.*?)</NATURAL>', structured_text, flags=_cl_re.S)
    stop_res_num = None
    if nat_match:
        nat_text = nat_match.group(1)
        stop_match = _cl_re.search(
            r'Reservoir\s+(\d+)\s*번째.*?(?:Stop\s*Solution|1\s*M\s*HCl|H2SO4|H₂SO₄)',
            nat_text, flags=_cl_re.I
        )
        if stop_match:
            stop_res_num = int(stop_match.group(1))

    if stop_res_num is not None:
        # In the sequence, if there is an ASPIRATE_FROM_PLATE(slot 1, plate) after the
        # Stop Solution reservoir (slot 5, col=stop_res_num) DISPENSE -> wash/removal after Stop = error
        stop_dispense_found = False
        for i, cmd in enumerate(commands):
            if not stop_dispense_found:
                # Find the pattern of ASPIRATE from the Stop Solution reservoir followed by DISPENSE
                if "ASPIRATE_FROM_PLATE" in cmd:
                    prm = params_list[i].split()
                    if len(prm) >= 2:
                        try:
                            if int(prm[0]) == 5 and int(prm[1]) == stop_res_num:
                                stop_dispense_found = True
                        except ValueError:
                            pass
            else:
                # Inspect commands after the Stop Solution dispense
                if "ASPIRATE_FROM_PLATE" in cmd:
                    prm = params_list[i].split()
                    if len(prm) >= 2:
                        try:
                            asp_slot = int(prm[0])
                            if asp_slot == 1:  # Aspirating from plate = solution removal or wash
                                issues.append({
                                    "expected": "After dispensing Stop Solution, perform reading only. No aspiration from the plate (solution removal/wash).",
                                    "observed": f"Plate aspiration found at index {i} after Stop Solution (Reservoir {stop_res_num}) dispense",
                                    "indices": [i],
                                })
                                break
                        except ValueError:
                            pass

    # -- (4) Minimum sequence row-count validation (code level -- independent of Claude/GPT/Llama) --
    dispense_count = sum(1 for c in commands if "DISPENSE_INTO_PLATE" in c)
    select_count = sum(1 for c in commands if "SELECT_CHANNEL" in c)

    # Determine the number of lines in structured [5]
    sec5_lines = 0
    inst_match2 = _cl_re.search(r'<INSTRUMENT>(.*?)</INSTRUMENT>', structured_text, flags=_cl_re.S)
    if inst_match2:
        sec5_match2 = _cl_re.search(r'\[5\](.*?)(?:\[6\]|$)', inst_match2.group(1), flags=_cl_re.S)
        if sec5_match2:
            sec5_lines = len([ln for ln in sec5_match2.group(1).strip().split('\n') if ln.strip()])

    # Minimum expected row count: structured [5] lines x 15, minimum 50 rows
    min_expected = max(sec5_lines * 15, 50)

    if total < min_expected:
        issues.append({
            "expected": (
                f"The sequence row count must be at least {min_expected} rows "
                f"(detected structured [5] {sec5_lines} lines, SELECT_CHANNEL {select_count} times, DISPENSE {dispense_count} times). "
                "[[5] format correction guide] Exact formats recognized by sequence_builder: "
                "(1) Type-based dispense: 'Reservoir N (타입명, 농도) 지정된 웰에 X µL/웰 분주' -- '지정된 웰에' is required. "
                "(2) Common reagent dispense: 'Reservoir N (시약명) X µL/웰 분주 (전체 웰)' or 'Reservoir N (시약명) 전체 웰에 X µL/웰 분주'. "
                "(3) Incubation: '상온(벤치탑) 배양 N분' or '37°C 배양 N분'. "
                "(4) Wash: 'Reservoir N (Wash Buffer)로 세척 N회, 각 X µL/웰'. "
                "(5) Solution removal: '(용액 제거) 이전 단계 용액 제거 X µL/웰'. "
                "Each line must exactly match one of the formats above for sequence_builder to generate the sequence. "
                "Lines that do not match the format are skipped, shortening the sequence."
            ),
            "observed": (
                f"Actual sequence row count: {total} rows -- suspected that sequence_builder skipped most lines due to format errors in [5] lines. "
                "Fix the format of each [5] line according to the guide above. Do not change the content (reagents, volumes, order)."
            ),
            "indices": [],
        })
    elif dispense_count == 0 and total > 0:
        issues.append({
            "expected": "The sequence must contain DISPENSE_INTO_PLATE commands (reagent dispensing is required)",
            "observed": (
                f"No DISPENSE_INTO_PLATE command in any of the {total} sequence rows -- "
                f"suspected sequence generation error (SELECT_CHANNEL {select_count} times)"
            ),
            "indices": [],
        })

    return issues


def _filter_false_positives(issues: list) -> list:
    """
    Remove, at the code level, false positives where Claude puts 'verified/correct' items into issues.
    If the observed field contains 'correct', 'no error', 'no actual error', etc., it is a
    verification log rather than an actual error, so remove it.
    """
    # Match with word-boundary patterns (e.g. prevent the bug of "correct" matching inside "incorrect")
    import re as _fp_re
    FALSE_POSITIVE_PATTERNS = [
        _fp_re.compile(r'\bis correct\b'),
        _fp_re.compile(r'\bcorrect\b(?!ed|ion|ly)'),  # 'correct' alone, excluding corrected/correction/correctly
        _fp_re.compile(r'\bno error\b'),
        _fp_re.compile(r'\bno actual error\b'),
        _fp_re.compile(r'\bno issue\b'),
        _fp_re.compile(r'\bthis is acceptable\b'),
        _fp_re.compile(r'\bnot an issue\b'),
        _fp_re.compile(r'\bnot an error\b'),
        _fp_re.compile(r'\bmatches\b'),
        _fp_re.compile(r'\bverified\b'),
        _fp_re.compile(r'\bas expected\b'),
        _fp_re.compile(r'\bno discrepancy\b'),
        _fp_re.compile(r'\bremoving this concern\b'),
        _fp_re.compile(r'\bno error here\b'),
        _fp_re.compile(r'\bdisregard\b'),
        _fp_re.compile(r'\ball correct\b'),
        _fp_re.compile(r'\bacceptable\b'),
        _fp_re.compile(r'\bactually present\b'),
        _fp_re.compile(r'\bis actually present\b'),
        _fp_re.compile(r'\bnot a real\b'),
        _fp_re.compile(r'\ball match\b'),
        _fp_re.compile(r'\btimes match\b'),
        _fp_re.compile(r'\bnot flagging\b'),
        _fp_re.compile(r'\bno clear error\b'),
    ]

    # Protected keywords for preparation-boundary issues that must never be removed
    PROTECTED_KEYWORDS = [
        "코팅", "봉쇄", "coating", "blocking", "block buffer",
        "capture ab", "plate preparation", "overinclusion",
        "[2] 사용자", "[5]에서 제거",
    ]

    filtered = []
    for issue in issues:
        if not isinstance(issue, dict):
            filtered.append(issue)
            continue

        observed = (issue.get("observed", "") or "").lower()
        expected = (issue.get("expected", "") or "").lower()
        combined_check = observed + " " + expected

        # -- Never remove issues containing a protected keyword --
        is_protected = any(pk in combined_check for pk in PROTECTED_KEYWORDS)
        if is_protected:
            filtered.append(issue)
            continue

        # -- Reservoir swap false-positive filter --
        is_reservoir_swap = (
            ("reservoir" in combined_check)
            and ("swap" in combined_check or "번호" in combined_check or "순서" in combined_check
                 or "column" in combined_check or "position" in combined_check
                 or "다르" in combined_check or "불일치" in combined_check
                 or "different" in combined_check or "mismatch" in combined_check)
            and ("wash" in combined_check or "diluent" in combined_check
                 or "buffer" in combined_check or "공통" in combined_check)
        )
        if is_reservoir_swap:
            continue

        # -- Shaking parameter false-positive filter --
        is_shaking_param = (
            ("shaking" in combined_check or "shake" in combined_check
             or "tap" in combined_check or "혼합" in combined_check)
            and ("rpm" in combined_check or "초" in combined_check or "second" in combined_check
                 or "시간" in combined_check or "duration" in combined_check)
        )
        if is_shaking_param:
            continue

        # Check whether observed/expected contains a false-positive pattern (word-boundary match)
        is_false_positive = False
        for pattern in FALSE_POSITIVE_PATTERNS:
            if pattern.search(combined_check):
                is_false_positive = True
                break

        if not is_false_positive:
            filtered.append(issue)

    return filtered


def _deduplicate_and_prioritize(issues: list) -> list:
    """
    Deduplicate issues + sort by severity.
    - For issues with identical observed text, keep only the first
    - Severity order: missing > incorrect/wrong > other
    """
    if not issues:
        return []

    # Deduplicate (based on observed text)
    seen = set()
    unique = []
    for issue in issues:
        if not isinstance(issue, dict):
            unique.append(issue)
            continue
        # Dedup key: normalized combination of observed + expected
        obs = (issue.get("observed", "") or "").strip().lower()
        exp = (issue.get("expected", "") or "").strip().lower()
        dedup_key = f"{obs}||{exp}"
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique.append(issue)

    # Assign severity scores (higher = more important)
    HIGH_PRIORITY_KEYWORDS = [
        "missing", "누락", "없", "not found", "not present",
        "코팅", "봉쇄", "coating", "blocking", "overinclusion",
    ]
    MEDIUM_PRIORITY_KEYWORDS = [
        "incorrect", "wrong", "잘못", "mismatch", "틀",
        "should be", "expected", "instead of",
    ]

    def severity_score(issue):
        if not isinstance(issue, dict):
            return 0
        combined = ((issue.get("observed", "") or "") + " " + (issue.get("expected", "") or "")).lower()
        score = 0
        for kw in HIGH_PRIORITY_KEYWORDS:
            if kw in combined:
                score = 2
                break
        if score == 0:
            for kw in MEDIUM_PRIORITY_KEYWORDS:
                if kw in combined:
                    score = 1
                    break
        return score

    # Sort by descending severity (higher score first)
    unique.sort(key=severity_score, reverse=True)
    return unique


def _read_rules_xlsx_as_text(xlsx_path) -> str:
    """
    Serialize the rules Excel content into a 'short text rule book' that Claude can easily understand.
    (Only the necessary columns, to prevent token explosion.)
    """
    if not xlsx_path or not os.path.exists(xlsx_path):
        return "(rules xlsx not found)"

    xl = pd.ExcelFile(xlsx_path)
    parts = []

    if "Rules" in xl.sheet_names:
        df = xl.parse("Rules")
        rows = []
        for _, r in df.iterrows():
            k = str(r.get("Key", "")).strip()
            v = str(r.get("Value", "")).strip()
            if k and v:
                rows.append(f"- {k}: {v}")
        parts.append("[RULES]\n" + "\n".join(rows[:200]))

    if "Deck_Map" in xl.sheet_names:
        df = xl.parse("Deck_Map")
        rows = []
        for _, r in df.iterrows():
            d = str(r.get("Deck_No", "")).strip()
            role = str(r.get("Role", "")).strip()
            if d and role:
                rows.append(f"- Deck {d}: {role}")
        parts.append("[DECK_MAP]\n" + "\n".join(rows[:200]))

    if "Compliance_Checklist" in xl.sheet_names:
        df = xl.parse("Compliance_Checklist")
        rows = []
        for _, r in df.iterrows():
            chk = str(r.get("Check", "")).strip()
            rule = str(r.get("Rule", "")).strip()
            if chk and rule:
                rows.append(f"- {chk}: {rule}")
        parts.append("[COMPLIANCE_CHECKLIST]\n" + "\n".join(rows[:200]))

    if "Command" in xl.sheet_names:
        df = xl.parse("Command")
        rows = []
        for _, r in df.iterrows():
            handler = str(r.get("Handler", "")).strip()
            cmd = str(r.get("Command", "")).strip()
            inputs = str(r.get("Inputs", "")).strip()
            if handler and cmd:
                rows.append(f"- H{handler} | {cmd} | inputs: {inputs}")
        parts.append("[ALLOWED_COMMANDS]\n" + "\n".join(rows[:300]))

    return "\n\n".join(parts)


def _sequence_df_to_dump(seq_df: pd.DataFrame, max_lines: int = 2500) -> str:
    """Convert Seq.xlsx into a text dump for Claude input."""
    lines = []
    for i, row in seq_df.iterrows():
        if i >= max_lines:
            lines.append(f"...(truncated, total_rows={len(seq_df)})")
            break
        h = str(row.get("Handler", "")).strip()
        c = str(row.get("Command", "")).strip()
        p = "" if pd.isna(row.get("Input Parameters", "")) else str(row.get("Input Parameters", "")).strip()
        lines.append(f"{i}: {h} | {c} | {p}")

    # -- Sequence fact summary (prevents Validator hallucination) --
    total = len(seq_df)
    disp_cols = set()  # params[1] = bioforge_col (column, 1~12)
    disp_rows = set()  # params[2] = bioforge_row (row, 1~8)
    for _, row in seq_df.iterrows():
        c = str(row.get("Command", "")).strip()
        if "DISPENSE_INTO_PLATE" in c:
            params = str(row.get("Input Parameters", "")).strip().split()
            if len(params) >= 3:
                try:
                    disp_cols.add(int(params[1]))
                    disp_rows.add(int(params[2]))
                except ValueError:
                    pass
    lines.append(f"\n[SEQUENCE FACTS — use these to verify your analysis]")
    lines.append(f"Total sequence rows: {total} (valid indices: 0 to {total - 1})")
    lines.append(f"DISPENSE params: (slot, bioforge_col, bioforge_row, volume, speed)")
    lines.append(f"  bioforge_col (2nd param, COLUMN 1~12) values used: {sorted(disp_cols)}")
    lines.append(f"  bioforge_row (3rd param, ROW 1~8) values used: {sorted(disp_rows)}")
    lines.append(f"Any index >= {total} does NOT exist. Do NOT reference non-existent indices.")
    lines.append(f"IMPORTANT: 2nd param is COLUMN (can be 1-12), NOT row. Only the 3rd param is ROW (max 8).")
    lines.append(f"If you see values 9-12 in the 2nd param, those are VALID column indices. Do NOT flag them as row errors.")

    return "\n".join(lines)


class ClaudeValidator:
    """
    Claude-only Validation Agent (individual call per stage)
    - Calls each of the 3 stages separately to ensure JSON parsing stability
    - Returns the aggregated final result
    """

    STAGE_CONFIGS = {
        "completeness": {
            "task": (
                "Stage: Completeness Check\n"
                "Check every action in ORIGINAL is reflected in STRUCTURED and realized in SEQUENCE.\n"
                "Example: 'Wash 4 times' => 4 wash cycles exist, each with aspirate/dispense/remove.\n"
                "Check removal steps exist before each new reagent addition.\n"
                "Check stop solution addition exists after TMB incubation.\n"
                "Note: 'gently tap'/'tap the plate' in the PDF = short shaking (SHAKE_START 200 10).\n"
                "If the sequence has a SHAKE_START where the PDF says 'tap', that counts as realized.\n"
                "\n"
                "SUBSTRATE / COLOR DEVELOPMENT INCUBATION (CRITICAL):\n"
                "- If PDF says 'incubate for color development' without specifying time,\n"
                "  the substrate type determines the expected incubation duration:\n"
                "  ABTS -> ~25 min (1500s), TMB -> ~20 min (1200s), pNPP -> ~30 min (1800s), OPD -> ~20 min (1200s).\n"
                "- WAIT_MOTION 60 (1 minute) for substrate color development is INCORRECT.\n"
                "  Substrate reactions require at least 10 minutes. Flag this as an error.\n"
                "- If PDF explicitly states a time, use that time.\n"
                "\n"
                "INCUBATION BLOCK RECOGNITION (CRITICAL):\n"
                "- After each dispense (4ch block + 1ch block), incubation is:\n"
                "  EJECT_PIPETTE -> TRANS 1 3 -> TEMPPLATE_ON 20 -> WAIT_MOTION N -> TRANS 3 1\n"
                "- This block appears RIGHT AFTER the last EJECT of a dispense step.\n"
                "  If you see TRANS/TEMPPLATE_ON/WAIT_MOTION/TRANS after EJECT, that IS the incubation.\n"
                "  Do NOT report 'incubation missing' if this pattern exists after the dispense block.\n"
                "\n"
                "PDF PAGE-BREAK DUPLICATION:\n"
                "- If the same reagent dispensing + incubation cycle appears twice consecutively\n"
                "  with identical reagent/volume/conditions, this is likely a PDF page-break artifact.\n"
                "  The duplicated cycle should be removed. Flag this if detected.\n"
                "\n"
                "BENCHTOP INCUBATION: When PDF says 'incubate on benchtop' or 'at room temperature' (without shaker),\n"
                "the correct automation is TRANS to Deck 3 (incubator) -> TEMPPLATE_ON 20 -> WAIT_MOTION -> TRANS back.\n"
                "Benchtop incubation uses Deck 3 at room temperature (20°C), NOT shaker. This is CORRECT.\n"
                "\n"
                "MISSING VALUE VERIFICATION — '?' AND '미기재' (CRITICAL):\n"
                "- If STRUCTURED_OUTPUT contains '?' or '미기재' for any parameter (volume, concentration, time,\n"
                "  dilution series, wash count, etc.), you MUST cross-check against the ORIGINAL_PROTOCOL PDF.\n"
                "- If the PDF DOES contain the value (even stated elsewhere in the document, in a table, figure,\n"
                "  reagent preparation section, or general instructions), this is an ERROR — GPT missed it.\n"
                "  Report: expected='[actual value from PDF]', observed='STRUCTURED has ?/미기재 but PDF states [value]'.\n"
                "- Common cases where GPT writes '?' or '미기재' but PDF actually has the info:\n"
                "  * Wash volume (e.g. PDF says '300 µL' in wash instructions, but STRUCTURED has '각 ? µL/웰')\n"
                "  * Standard dilution series concentrations (in a table, typical data, or figure)\n"
                "  * Incubation times (in general procedure or step-by-step instructions)\n"
                "  * Substrate type (inferrable from kit name, reagent list, or materials provided)\n"
                "  * Sample volume (stated in assay procedure)\n"
                "- If the PDF genuinely does NOT contain the value, '?' or '미기재' is CORRECT — do NOT flag it.\n"
                "- Check EVERY '?' and '미기재' in STRUCTURED against the PDF. Do not skip any.\n"
                "\n"
                "PLATE PREPARATION BOUNDARY CHECK (CRITICAL — OVERINCLUSION DETECTION):\n"
                "Coating/blocking (Plate Preparation) must NEVER appear in the [5] automation!\n"
                "- Capture Antibody coating (dispense + overnight/multi-hour incubation) must appear ONLY in the [2] user manual steps.\n"
                "- Blocking (Block Buffer dispense + incubation) must also appear ONLY in [2].\n"
                "- The wash after coating and the wash after blocking must also appear only in [2].\n"
                "- If any coating/blocking step is included in the [5] automation, report FAIL!\n"
                "  expected='Coating/blocking must appear only in the [2] user steps'\n"
                "  observed='[5] contains a coating/blocking step: [quote the relevant line]'\n"
                "- If the [4] Reservoir mapping contains Capture Ab or Block Buffer, FAIL!\n"
                "  (The user performs it manually, so no Reservoir is needed)\n"
                "- Apply this rule regardless of kit type (DuoSet, Development Kit, etc.).\n"
                "- Starting point of [5]: from the first dispense after coating/blocking is complete (Standard/Sample/Assay Diluent, etc.).\n"
                "\n"
                "[5] FIRST LINE VALIDATION (CRITICAL):\n"
                "The first line of [5] must be a Blank or the first type-based Reservoir dispense!\n"
                "- Valid [5] start: 'Reservoir 1 (Blank ...) 지정된 웰에 X µL/웰 분주'\n"
                "- Invalid [5] start: '(용액 제거)...', '세척...', 'Wash Buffer...' -> FAIL!\n"
                "- Invalid [5] start: starting from Reservoir 2 onward with Reservoir 1 (Blank) missing -> FAIL!\n"
                "  If PLATE_LAYOUT has a Blank, the first dispense line of [5] must be Blank.\n"
                "- Invalid [5] start: a common reagent such as 'Diluent' or 'Assay Diluent' dispensed on the first line without '지정된 웰에' -> FAIL!\n"
                "  Having Diluent on the first line of [5] as a common reagent (not type-based, designated wells) is an incorrect structure.\n"
                "- If [5] starts with solution removal or wash, the last step of blocking (solution removal + wash)\n"
                "  has been incorrectly carried over from [2] into [5]. -> Remove that line from [5] and include it in [2].\n"
                "- This error can occur even without keywords like '코팅' or 'block buffer'!\n"
                "  Judge by context: if [2] has blocking and the first line of [5] is solution removal/wash -> boundary error.\n"
                "\n"
                "OVERINCLUSION DETECTION (CRITICAL — NEW):\n"
                "Not only 'what is missing' but also 'what was added that is not in the PDF' must be caught!\n"
                "- GPT sometimes invents and adds steps that are not in the PDF (hallucination).\n"
                "- Compare every step in STRUCTURED [5] against the ORIGINAL_PROTOCOL PDF.\n"
                "- If a step not described in the PDF is included in [5], report FAIL!\n"
                "  expected='Steps not in the PDF must not be included in [5]'\n"
                "  observed='[5] contains a step not in the PDF: [quote the relevant line]'\n"
                "- Patterns to watch especially:\n"
                "  * Solution removal/wash added after Stop Solution (not in PDF -> destroys the experiment)\n"
                "  * Coating/blocking steps duplicated into [5]\n"
                "  * Extra incubation/wash cycles repeated that are not in the PDF\n"
                "  * Dispense steps added for reagents not mentioned in the PDF\n"
                "- If the [5] automation block includes any of the following, FAIL:\n"
                "  * Capture Antibody coating dispense\n"
                "  * Block Buffer dispense\n"
                "  * Overnight/long incubation after coating (room-temperature overnight, 4°C overnight, etc.)\n"
                "  * Wash immediately after coating/blocking\n"
                "- If the [4] Reservoir mapping includes any of the following, FAIL:\n"
                "  * Capture Antibody (for coating)\n"
                "  * Block Buffer / Blocking Buffer\n"
                "- If [5] of STRUCTURED_OUTPUT contains keywords such as '코팅', 'coating', 'Capture Ab 분주', 'Block Buffer 분주',\n"
                "  you must catch them.\n"
                "\n"
                "STOP SOLUTION POST-STEP CHECK (CRITICAL — ASSAY-DESTROYING ERROR):\n"
                "After dispensing Stop Solution, solution removal or wash must NEVER be performed!\n"
                "- Stop Solution (e.g. 1M HCl, 2N H₂SO₄) is the last reagent of ELISA.\n"
                "- After dispensing Stop Solution, you must proceed directly to reading ([6]).\n"
                "- If [5] has '용액 제거', '세척', 'Wash', etc. after the Stop Solution dispense, FAIL!\n"
                "  expected='After dispensing Stop Solution, perform reading only. No solution removal/wash.'\n"
                "  observed='A solution removal/wash step is included after Stop Solution'\n"
                "- Adding Stop Solution and then washing removes all of the reaction product, destroying the experiment.\n"
                "- Check the SEQUENCE too: if there is an ASPIRATE_FROM_PLATE (aspirate from well) or a wash block\n"
                "  after the Stop Solution Reservoir dispense command, FAIL.\n"
                "\n"
                "IMPORTANT — DEFAULTS vs PDF-specified values:\n"
                "RULES defaults are FALLBACKS for when the PDF does not specify a value.\n"
                "If the PDF explicitly states a value, that value is CORRECT even if it differs from defaults.\n"
                "\n"
                "DO NOT FLAG these as errors:\n"
                "- TIP POSITIONS ARE MANAGED BY THE SYSTEM — DO NOT VALIDATE TIP COORDINATES.\n"
                "  4ch and 1ch share a single tip allocator with a shared 'used' set.\n"
                "  When 1ch uses positions between 4ch allocations, 4ch will skip those positions.\n"
                "  Example: 4ch uses (1,1)->(1,5), then 1ch uses (2,1),(2,2)... -> next 4ch skips to (3,1) or (2,5).\n"
                "  This is CORRECT — positions appear 'out of order' because they are shared.\n"
                "- PLATE COORDINATES: ASPIRATE/DISPENSE params are (slot, bioforge_col, bioforge_row, vol, speed).\n"
                "  2nd param = COLUMN (1~12), 3rd param = ROW (1~8). Col 9-12 are VALID. Only flag row > 8.\n"
                "- INSTALL_PIPETTE coordinates (tip_slot, tip_row, tip_col) are TIP RACK positions, NOT plate positions.\n"
                "  Tip rows can be 9, 10, 11, 12+ — this does NOT mean the plate has rows 9~12.\n"
                "  Only ASPIRATE_FROM_PLATE/DISPENSE_INTO_PLATE use plate coordinates.\n"
                "  Do NOT flag tip position order, gaps, or skips. The system handles this automatically.\n"
                "- Tip position REUSE is NORMAL. The tip rack has limited capacity.\n"
                "  When all positions are exhausted, the system reuses positions from the beginning.\n"
                "  The operator replaces physical tips during operation. Do NOT flag tip reuse.\n"
                "- 4ch/1ch DISPENSING STRUCTURE: Each dispense step is split into 4ch block + 1ch block.\n"
                "  Incubation appears AFTER both blocks. Do NOT flag this as 'interleaved' or 'misplaced'.\n"
                "- TYPE-BASED RESERVOIR DISPENSING: Reservoirs 1~N (Blank/Calibrant/Sample) each dispense to ONLY their designated wells.\n"
                "  In [5], each type-based Reservoir appears as exactly ONE line per Reservoir: 'Reservoir N (...) 지정된 웰에 X µL/웰 분주.'\n"
                "  The system handles replicates/triplicates automatically via well coordinates from the UI plate layout.\n"
                "  Each Reservoir line generates dispense commands for ALL wells assigned to that Reservoir (including replicates).\n"
                "  Do NOT flag 'only 1 dispense line per Reservoir' as missing replicates — replicates ARE included automatically.\n"
                "  In the sequence, each type-based Reservoir aspirates from slot 5 position (N,1) and targets only a SUBSET of wells.\n"
                "  This is CORRECT — do NOT flag as 'only dispensing to some wells' or 'not dispensing to all wells'.\n"
                "  The number of dispense commands per Reservoir depends on how many wells were assigned to it in the UI (not on protocol text).\n"
                "  Common reagent Reservoirs (Wash, Conjugate, Substrate, Stop) DO dispense to ALL wells.\n"
                "- Steps in [1]-[3] that are PREPARATION/DILUTION (not dispensing) are manual — exclude from [5].\n"
                "  However, ALL DISPENSING steps ('add to wells', 'pipette into wells', '분주') MUST be in [5]!\n"
                "  This system has automated dispensing for ALL solution types including Standard/Sample/Blank.\n"
                "  If PDF says 'add Assay Diluent to wells' or 'add sample to wells', it MUST appear in [5].\n"
                "  Flag as MISSING if any dispensing-to-well step is placed in [3] instead of [5].\n"
                "- CONTROL IS NOT A SEPARATE RESERVOIR TYPE.\n"
                "  This system only has 3 type-based Reservoir types: Blank, Calibrant(Standard), Sample.\n"
                "  PDF may mention 'control' alongside 'standard, control, or sample', but Control does NOT get its own Reservoir.\n"
                "  Do NOT flag missing Control Reservoir or missing Control dispense line. Control is excluded by design.\n"
                "- USER-DEFINED CALIBRANT OVERFLOW IS ALLOWED.\n"
                "  When the plate_mapper allocates more Calibrant Reservoirs than the PDF defines standards,\n"
                "  the extra Reservoirs are labeled '사용자 지정 Calibrant (User-defined)'. This is CORRECT.\n"
                "  Do NOT flag 'PDF has no 9th calibrant' — the user intentionally added extra calibrant points.\n"
                "- RESERVOIR NUMBER CONTINUITY CHECK:\n"
                "  Type-based Reservoirs (Blank/Calibrant/Sample) MUST be numbered consecutively from 1.\n"
                "  If plate_mapper allocated Reservoirs 1~13, then ALL numbers 1 through 13 must appear.\n"
                "  A GAP (e.g. 1~8 then 10~13, skipping 9) means a Reservoir was DROPPED — flag as FAIL.\n"
                "  Check BOTH [4] Reservoir mapping AND [5] automation dispense lines for completeness.\n"
                "\n"
                "PLATE_LAYOUT VERIFICATION (CRITICAL — IF [PLATE_LAYOUT] IS PROVIDED):\n"
                "If a [PLATE_LAYOUT] section is present, you must verify the following!\n"
                "- [PLATE_LAYOUT] specifies the exact well configuration used in this experiment:\n"
                "  how many Blanks, how many Calibrants (Standards), how many Samples, and up to which Reservoir number.\n"
                "- The STRUCTURED [4] Reservoir mapping must include all types (Blank/Calibrant/Sample) from PLATE_LAYOUT.\n"
                "  Example: if PLATE_LAYOUT has 4 Samples (Reservoir 10~13), then [4] must have 4 Samples.\n"
                "  If [4] has only 1 Sample -> FAIL: 'PLATE_LAYOUT has N Samples but [4] has only M'\n"
                "- The STRUCTURED [5] automation must also have dispense lines for all types in PLATE_LAYOUT.\n"
                "  Example: if PLATE_LAYOUT has 4 Samples, [5] must have all of 'Reservoir 10', 'Reservoir 11', 'Reservoir 12', 'Reservoir 13' dispenses.\n"
                "  If some Samples are missing from [5] -> FAIL: 'Of the N Samples per PLATE_LAYOUT, only M are in [5]'\n"
                "- The [NEXT_AVAILABLE_RESERVOIR] tag indicates the starting Reservoir number of common reagents.\n"
                "  All Reservoirs before this number (1 ~ N-1) must be Blank/Calibrant/Sample.\n"
                "  In [4], the number of Reservoirs in this range must match the per-type counts in PLATE_LAYOUT.\n"
                "- SHAKE_START (rpm, seconds) ALREADY INCLUDES the wait time. The second parameter IS the duration.\n"
                "  e.g. SHAKE_START 500 3600 = shake at 500 rpm for 3600 seconds. No separate WAIT_MOTION needed.\n"
                "  Do NOT flag missing WAIT_MOTION after SHAKE_START — it is BUILT INTO the command.\n"
                "\n"
                "WASH BLOCK TIP POLICY (TIP CONTAMINATION PREVENTION):\n"
                "- In a wash block the tip touches the well (during aspiration), so the tip must be changed per group/well.\n"
                "  Correct wash pattern (per-round 4ch/1ch alternation):\n"
                "  [repeat N rounds]: 4ch round 1 (each group: INSTALL->work->EJECT) -> 1ch round 1 (each well: INSTALL->work->EJECT)\n"
                "  i.e. 4ch wash round 1 -> 1ch wash round 1 -> 4ch wash round 2 -> 1ch wash round 2 -> ... (uniform timing for all wells)\n"
                "- In a solution removal (remove) block the tip also touches the well, so a per-group/per-well tip change is required.\n"
                "  Correct 4ch removal pattern: [each group]: SELECT_4CH -> INSTALL -> well_aspirate->waste -> EJECT\n"
                "  Correct 1ch removal pattern: [each well]: SELECT_1CH -> INSTALL -> well_aspirate->waste -> EJECT\n"
                "- Dispense does not touch the well, so tip reuse is OK when dispensing from the same Reservoir.\n"
                "  Correct dispense pattern: INSTALL once -> [all groups/wells: reservoir_aspirate->well_dispense] -> EJECT once\n"
                "- A channel switch (4ch->1ch, 1ch->4ch) also requires EJECT -> new INSTALL.\n"
                "- A high frequency of tip EJECT/INSTALL (changing per group/well) is normal and not an error.\n"
            ),
            "schema": (
                'Return JSON ONLY: {"result":"PASS or FAIL","issues":[{"expected":"...","observed":"...","indices":[]}]}\n'
                "CRITICAL RULES:\n"
                "- ONLY include actual errors in issues. If something is CORRECT, do NOT add it to issues.\n"
                "- If there are NO errors, return {\"result\":\"PASS\",\"issues\":[]}.\n"
                "- issues list must contain ONLY genuinely wrong/missing items.\n"
                "FEEDBACK QUALITY (CRITICAL for regeneration):\n"
                "- 'expected' field: be SPECIFIC about exact location and scope of the fix.\n"
                "  * Reservoir mapping issue: name the EXACT section (e.g. '[4] Reservoir mapping'),\n"
                "    EXACT item (e.g. 'Reservoir 17th: Block Buffer (1% BSA in PBS, 300μL/well)'),\n"
                "    WHERE it should go instead (e.g. 'list only in the [2] user manual steps, no Reservoir needed').\n"
                "    End with: 'Do not change the other sections ([1]~[3], [5], etc.) or other reagents apart from this item.'\n"
                "  * Missing step: specify exact step name/position to add, not just 'add it'.\n"
                "- 'observed' field: state only the conclusion, no reasoning process.\n"
                "- Vague 'expected' like 'should not be in reservoir' causes GPT to over-correct\n"
                "  and delete far more than intended. Always specify the MINIMUM change needed.\n"
                "- Do NOT list items you verified as correct. Correct items are NOT issues.\n"
                "- If your analysis concludes 'acceptable', 'correct', 'not an error', 'realized',\n"
                "  'within range', or 'per conventions' — that means it is NOT an issue. Do NOT add it.\n"
                "- Efficiency concerns (e.g. unnecessary TRANS round-trips) are NOT errors. Ignore them.\n"
                "Keep each issue SHORT. Maximum 5 issues. No prose outside JSON."
            ),
        },
        "parameter_accuracy": {
            "task": (
                "Stage: Parameter Accuracy Check\n"
                "Check numeric and mapping correctness:\n"
                "- 100 uL => volume=100\n"
                "- 2 hours => 7200 seconds, 30 min => 1800 seconds\n"
                "- RT (room temperature) => use Default_RT_Temperature from RULES (NOT hardcoded 25C)\n"
                "- Reservoir mapping: use the STRUCTURED_OUTPUT's [4] Reservoir mapping as the AUTHORITATIVE source.\n"
                "  The STRUCTURED_OUTPUT defines which Reservoir number = which reagent.\n"
                "  Sequence slot (5,N,1) corresponds to Reservoir N. Verify against STRUCTURED mapping ONLY.\n"
                "- RESERVOIR NUMBERING: This system uses TWO ranges of Reservoir numbers, ALL listed in [4]:\n"
                "  1) Low numbers (1~N): Type-based dispensing (Blank/Calibrant/Sample). Each type gets its own Reservoir.\n"
                "     These appear in [4] with PDF-derived descriptions (e.g. 'Blank (Diluent, 0 pg/mL)', 'Standard S1 (200 pg/mL)').\n"
                "     In [5], each type-based Reservoir appears as exactly ONE line: 'Reservoir N (...) 지정된 웰에 X µL/웰 분주.'\n"
                "     The system uses UI well-plate drag info to dispense each Reservoir to ONLY its designated wells (not all wells).\n"
                "     Replicates/triplicates are handled AUTOMATICALLY by the system via well coordinates — NOT by repeating lines.\n"
                "     In sequence, each type-based Reservoir aspirates from slot 5 position (N,1) and dispenses ONLY to its mapped wells.\n"
                "     The number of dispense commands depends on the UI well layout, not on the number of lines in [5].\n"
                "     Do NOT flag 'only 1 line per Reservoir' as missing replicates. Do NOT flag 'fewer dispenses than expected'.\n"
                "  2) High numbers (N+1~): Common reagents (Wash, Assay Diluent, Conjugate, Substrate, Stop, etc.).\n"
                "     These also appear in [4] and dispense to ALL wells on the plate.\n"
                "  ALL Reservoirs (type-based + common) are listed in [4]. Verify Reservoir numbers against [4] mapping.\n"
                "- TIP CONTAMINATION POLICY:\n"
                "  * WASH/REMOVE blocks: tips TOUCH wells during aspiration -> tip change per group (4ch) / per well (1ch).\n"
                "    Wash (round-robin with 4ch/1ch alternation): [each round]: 4ch round (each group: INSTALL->work->EJECT) -> 1ch round (each well: INSTALL->work->EJECT).\n"
                "    i.e. 4ch wash round 1 -> 1ch wash round 1 -> 4ch wash round 2 -> 1ch wash round 2 -> ...\n"
                "    Remove: [each group/well]: INSTALL -> aspirate_well->waste -> EJECT.\n"
                "  * DISPENSE blocks: tips do NOT touch wells -> same tip can be reused for all groups/wells of the same Reservoir.\n"
                "    Dispense: INSTALL -> [all groups/wells: aspirate_reservoir->dispense_well] -> EJECT.\n"
                "  * Frequent INSTALL/EJECT in wash/remove blocks is CORRECT, not an error.\n"
                "- TIP POSITIONS — DO NOT VALIDATE. 4ch and 1ch share one allocator with shared 'used' set.\n"
                "  Positions may appear out of order or have gaps because 1ch occupies positions between 4ch.\n"
                "  Do NOT flag tip position order, gaps, skips, or reuse. System manages this automatically.\n"
                "- PLATE COORDINATES: ASPIRATE/DISPENSE params are (slot, bioforge_col, bioforge_row, vol, speed).\n"
                "  2nd param = COLUMN (1~12), 3rd param = ROW (1~8). Col 9-12 are VALID. Only flag row > 8.\n"
                "- INSTALL_PIPETTE coordinates (tip_slot, tip_row, tip_col) are TIP RACK positions, NOT plate positions.\n"
                "  Tip rows can be 9, 10, 11, 12+ — this does NOT mean the plate has rows 9~12.\n"
                "  Only ASPIRATE_FROM_PLATE/DISPENSE_INTO_PLATE use plate coordinates.\n"
                "- 4ch/1ch DISPENSING STRUCTURE: Each dispense step produces 4ch block + 1ch block in sequence.\n"
                "  Incubation appears AFTER both blocks. Do NOT flag this as 'interleaved' or 'misplaced'.\n"
                "\n"
                "SUBSTRATE / COLOR DEVELOPMENT INCUBATION (CRITICAL):\n"
                "- Substrate incubation (ABTS, TMB, pNPP, OPD) for color development:\n"
                "  If PDF says 'incubate for color development' without explicit time:\n"
                "  ABTS -> WAIT_MOTION should be ~1500 (25 min), NOT 60.\n"
                "  TMB  -> WAIT_MOTION should be ~1200 (20 min), NOT 60.\n"
                "  pNPP -> WAIT_MOTION should be ~1800 (30 min), NOT 60.\n"
                "  If WAIT_MOTION is 60s for substrate incubation, this is WRONG — flag it!\n"
                "  In your feedback, specify the EXACT correct value (e.g. 'WAIT_MOTION should be 1500, not 60').\n"
                "- If PDF explicitly states incubation time, use that time.\n"
                "\n"
                "INCUBATION BLOCK RECOGNITION (CRITICAL — READ CAREFULLY):\n"
                "- After dispensing blocks (4ch + 1ch), the incubation block is:\n"
                "  EJECT -> TRANS 1 3 -> TEMPPLATE_ON 20 -> WAIT_MOTION N -> TRANS 3 1\n"
                "- This pattern appears IMMEDIATELY after the last EJECT_PIPETTE of a dispense step.\n"
                "  The TRANS/TEMPPLATE_ON/WAIT_MOTION/TRANS IS the incubation for the preceding dispense.\n"
                "- When checking 'is incubation present after substrate dispense?', look at the rows\n"
                "  RIGHT AFTER the final EJECT_PIPETTE of that dispense block. If TRANS 1 3 / TEMPPLATE_ON /\n"
                "  WAIT_MOTION / TRANS 3 1 follows, the incubation EXISTS. Do NOT report it as missing.\n"
                "- Count carefully: the EJECT at the end of 1ch block -> next row is TRANS 1 3 -> that IS incubation.\n"
                "\n"
                "PDF PAGE-BREAK DUPLICATION:\n"
                "- If the SAME substrate dispensing + incubation cycle is duplicated back-to-back\n"
                "  (identical reagent, volume, incubation), flag the duplicate for removal.\n"
                "\n"
                "MISSING VALUE VERIFICATION — '?' AND '미기재' (CRITICAL):\n"
                "- If STRUCTURED_OUTPUT contains '?' or '미기재' for any parameter (volume, concentration, time,\n"
                "  dilution series, wash count, etc.), you MUST cross-check against the ORIGINAL_PROTOCOL PDF.\n"
                "- If the PDF DOES contain the value (even stated elsewhere — in tables, figures, reagent prep,\n"
                "  general instructions, or materials sections), this is an ERROR — GPT missed it.\n"
                "  Report: expected='[actual value from PDF]', observed='STRUCTURED has ?/미기재 but PDF states [value]'.\n"
                "- Common cases where GPT writes '?' or '미기재' but PDF actually has the info:\n"
                "  * Wash volume (e.g. '300 µL' in wash instructions -> '각 ? µL/웰' is wrong)\n"
                "  * Standard dilution concentrations (in typical data table or figure)\n"
                "  * Incubation times (in procedure steps or general instructions)\n"
                "  * Substrate type (from kit name, reagent list, or materials provided)\n"
                "- If the PDF genuinely does NOT contain the value, '?' or '미기재' is CORRECT — do NOT flag it.\n"
                "- Check EVERY '?' and '미기재' in STRUCTURED against the PDF. Do not skip any.\n"
                "\n"
                "PLATE PREPARATION BOUNDARY CHECK (CRITICAL — OVERINCLUSION DETECTION):\n"
                "Coating/blocking (Plate Preparation) must NEVER appear in the [5] automation!\n"
                "- If STRUCTURED [5] includes any coating/blocking step (Capture Ab coating, Block Buffer dispense, wash after coating, etc.),\n"
                "  FAIL. Coating/blocking must exist only in the [2] user manual steps.\n"
                "- If the [4] Reservoir mapping has Capture Ab or Block Buffer, FAIL.\n"
                "  (The user performs it manually, so it must not be mapped to a Reservoir)\n"
                "- If the SEQUENCE has commands corresponding to coating/blocking, FAIL:\n"
                "  e.g. a sequence that aspirates Capture Ab from a Reservoir and dispenses it into wells, a dispense from a Block Buffer Reservoir, etc.\n"
                "- Starting point of [5]: from the first dispense after coating/blocking is complete (Standard/Sample/Assay Diluent, etc.).\n"
                "- Apply regardless of kit type (DuoSet, Development Kit, etc.).\n"
                "\n"
                "CRITICAL — DEFAULTS vs PDF-specified values:\n"
                "The RULES section contains DEFAULT values (shake_speed, shake_time_sec, volume_ul, etc.).\n"
                "These defaults are FALLBACKS used ONLY when the original PDF does NOT specify a value.\n"
                "If the ORIGINAL_PROTOCOL PDF explicitly states a value (e.g. '500 rpm', '400 µL', '60 min'),\n"
                "then the PDF-specified value OVERRIDES the default and MUST be used in the sequence.\n"
                "Do NOT flag a parameter as wrong just because it differs from the RULES default.\n"
                "Compare parameters against the ORIGINAL_PROTOCOL PDF first, RULES defaults second.\n"
                "Example: RULES has shake_speed=5, but PDF says '500 ± 50 rpm'\n"
                "  => SHAKE_START 500 is CORRECT (PDF overrides default). Do NOT report as error.\n"
                "\n"
                "AUTOMATION CONVENTIONS (do NOT flag as errors):\n"
                "- 'Gently tap' / 'tap the plate' = SHAKE_START 200 N (200 rpm, N seconds).\n"
                "  If PDF specifies tap duration (e.g. '~1 minute'), N = that duration. CORRECT by design.\n"
                "- SHAKE_START (rpm, seconds): the second parameter IS the duration.\n"
                "  e.g. SHAKE_START 500 3600 = shake for 3600 sec. No separate WAIT_MOTION needed after it.\n"
                "- Benchtop / RT incubation: plate is transferred to Deck 3 (incubator) at 20°C.\n"
                "  TRANS 1->3, TEMPPLATE_ON 20, WAIT_MOTION, TRANS 3->1 is the CORRECT pattern.\n"
                "  In this automation system, ALL incubation (including benchtop/RT) uses Deck 3.\n"
                "  The plate does NOT stay on Deck 1. Do NOT flag Deck 3 transfer as wrong for benchtop.\n"
                "\n"
                "WASH/REMOVE TIP POLICY (TIP CONTAMINATION PREVENTION):\n"
                "- WASH and REMOVE blocks: tips TOUCH wells during aspiration -> tip must be changed per group (4ch) / per well (1ch).\n"
                "  Wash (round-robin, 4ch/1ch alternation): [each round]: 4ch round (each group: INSTALL->work->EJECT) -> 1ch round (each well: INSTALL->work->EJECT).\n"
                "  i.e. 4ch wash round 1 -> 1ch wash round 1 -> 4ch wash round 2 -> 1ch wash round 2 -> ...\n"
                "  Remove: [each group/well]: INSTALL -> well_aspirate->waste -> EJECT.\n"
                "  This means many INSTALL/EJECT pairs per wash/remove block. This is CORRECT.\n"
                "- DISPENSE blocks: tips do NOT touch wells -> same tip reused for all groups/wells of same Reservoir. CORRECT.\n"
                "- Channel switch (4ch->1ch, 1ch->4ch) also requires EJECT->INSTALL.\n"
                "- Do NOT flag frequent tip changes in wash/remove as errors — they are required for contamination prevention.\n"
                "\n"
                "STOP SOLUTION POST-STEP CHECK (CRITICAL — ASSAY-DESTROYING ERROR):\n"
                "After dispensing Stop Solution, there must NEVER be solution removal or wash!\n"
                "- Stop Solution (1M HCl, 2N H₂SO₄, etc.) is the last reagent of ELISA.\n"
                "- After dispensing Stop Solution -> perform reading ([6]) only. Solution removal/wash loses the results.\n"
                "- If STRUCTURED [5] has '용액 제거', '세척', or 'Wash' after the Stop Solution line, FAIL!\n"
                "- If the SEQUENCE has an ASPIRATE_FROM_PLATE after the Stop Solution Reservoir dispense, FAIL!\n"
                "- This is a step not in the PDF that GPT added as a hallucination.\n"
                "\n"
                "OVERINCLUSION DETECTION (detecting steps added that are not in the PDF):\n"
                "- Compare every step in STRUCTURED [5] against the ORIGINAL_PROTOCOL PDF.\n"
                "- If a step not described in the PDF is added to [5], FAIL!\n"
                "- Especially: wash after Stop, extra wash/incubation cycles not in the PDF, dispensing of reagents not mentioned.\n"
                "\n"
                "CONTROL AND USER-DEFINED CALIBRANT POLICY:\n"
                "- CONTROL IS NOT A SEPARATE RESERVOIR TYPE. Only Blank/Calibrant/Sample exist.\n"
                "  PDF 'control' is excluded by design. Do NOT flag missing Control Reservoir or dispense.\n"
                "- USER-DEFINED CALIBRANT OVERFLOW IS ALLOWED.\n"
                "  Extra Calibrant Reservoirs beyond PDF standard count are labeled '사용자 지정 Calibrant (User-defined)'.\n"
                "  This is intentional. Do NOT flag as 'PDF has no Nth calibrant'.\n"
                "- RESERVOIR NUMBER CONTINUITY: Type-based Reservoirs must be consecutively numbered from 1.\n"
                "  A gap (e.g. skipping Reservoir 9) means a dispense step was DROPPED. Flag as FAIL.\n"
            ),
            "schema": (
                'Return JSON ONLY: {"result":"PASS or FAIL","issues":[{"expected":"...","observed":"...","indices":[]}]}\n'
                "CRITICAL RULES:\n"
                "- ONLY include actual errors in issues. If a parameter is CORRECT, do NOT add it to issues.\n"
                "- If all parameters are correct, return {\"result\":\"PASS\",\"issues\":[]}.\n"
                "- issues list must contain ONLY genuinely wrong values.\n"
                "- Do NOT list items you verified as correct. Verified-correct items are NOT issues.\n"
                "- 'Correct' or 'matches' means it is NOT an issue — do NOT include it.\n"
                "- If your analysis concludes 'acceptable', 'correct', 'within range', 'per conventions',\n"
                "  or 'not an error' — that is NOT an issue. Do NOT add it to the issues list.\n"
                "Keep each issue SHORT. Maximum 5 issues. No prose outside JSON."
            ),
        },
        "execution_order": {
            "task": (
                "Stage: Execution Order Check\n"
                "Physical constraints:\n"
                "- Tip install MUST come before aspirate/dispense, eject after use.\n"
                "- Plate must be transported to correct device before device actions.\n"
                "- Channel mode (4-ch vs 1-ch) must be set before pipette operations.\n"
                "- TIP POSITIONS — DO NOT VALIDATE ORDER, GAPS, OR SKIPS.\n"
                "  4ch and 1ch share a single allocator. 1ch may occupy positions between 4ch allocations,\n"
                "  causing 4ch to skip positions. This is CORRECT. Tip reuse is also NORMAL after exhaustion.\n"
                "  Do NOT flag tip position order, gaps, skips, reuse, or 'already consumed' positions.\n"
                "- PLATE COORDINATES: ASPIRATE/DISPENSE params are (slot, bioforge_col, bioforge_row, vol, speed).\n"
                "  2nd param = COLUMN (1~12), 3rd param = ROW (1~8). Col 9-12 are VALID. Only flag row > 8.\n"
                "- INSTALL_PIPETTE coordinates (tip_slot, tip_row, tip_col) are TIP RACK positions, NOT plate positions.\n"
                "  Tip rows can be 9, 10, 11, 12+ — this does NOT mean the plate has rows 9~12.\n"
                "  Only ASPIRATE_FROM_PLATE/DISPENSE_INTO_PLATE use plate coordinates.\n"
                "- Flag any impossible ordering (but NOT tip position ordering).\n"
                "\n"
                "AUTOMATION CONVENTIONS (do NOT flag as errors):\n"
                "- ALL incubation (including benchtop/RT) uses Deck 3: TRANS 1->3, TEMPPLATE_ON, WAIT_MOTION, TRANS 3->1.\n"
                "  Plate does NOT stay on Deck 1. Deck 3 at 20°C for benchtop is CORRECT.\n"
                "- 'Gently tap' = SHAKE_START 200 N. SHAKE_START includes duration (2nd param = seconds).\n"
                "  No separate WAIT_MOTION needed after SHAKE_START.\n"
                "- 4ch/1ch DISPENSING STRUCTURE: Each dispense step is split into 4ch block + 1ch block.\n"
                "  Incubation appears AFTER both 4ch and 1ch blocks complete. This is the correct order:\n"
                "  4ch dispense -> 1ch dispense -> incubation -> next step.\n"
                "  Do NOT flag incubation as 'interleaved between 4ch and 1ch' or 'misplaced'.\n"
                "\n"
                "SUBSTRATE / COLOR DEVELOPMENT INCUBATION:\n"
                "- After substrate dispensing (ABTS/TMB/pNPP/OPD), there must be an incubation step.\n"
                "  The incubation should be at least 10 minutes (600s). WAIT_MOTION 60 is too short.\n"
                "- After substrate incubation, a Stop Solution step should typically follow (if protocol specifies).\n"
                "  Check ordering: Substrate dispense -> Incubation -> Stop Solution -> Read.\n"
                "\n"
                "WASH/REMOVE TIP POLICY (TIP CONTAMINATION PREVENTION):\n"
                "- WASH blocks: tips touch wells during aspiration -> per-group (4ch) / per-well (1ch) tip change.\n"
                "  Round-robin with 4ch/1ch alternation: [each round]: 4ch (each group: INSTALL->work->EJECT) -> 1ch (each well: INSTALL->work->EJECT).\n"
                "  i.e. 4ch wash round 1 -> 1ch wash round 1 -> 4ch wash round 2 -> 1ch wash round 2 -> ...\n"
                "  This creates many INSTALL/EJECT pairs. This is the CORRECT pattern.\n"
                "- REMOVE blocks: tips touch wells -> per-group (4ch) / per-well (1ch) tip change.\n"
                "  Pattern: [each group/well]: SELECT -> INSTALL -> well_aspirate -> waste -> EJECT.\n"
                "- DISPENSE blocks: tips do NOT touch wells -> single INSTALL, multiple dispenses, single EJECT.\n"
                "  Pattern: SELECT -> INSTALL -> [all groups/wells: reservoir_aspirate -> well_dispense] -> EJECT.\n"
                "- Channel switch (4ch->1ch, 1ch->4ch) also requires EJECT->INSTALL.\n"
                "- Do NOT flag frequent tip EJECT/INSTALL in wash/remove as ordering errors — they are contamination prevention.\n"
                "\n"
                "- The SEQUENCE covers [5] automation steps. Preparation/dilution from [1]-[3] is done BY HAND.\n"
                "  However, ALL dispensing (including Assay Diluent, Standard, Sample, Blank) MUST be in [5].\n"
                "  The automation starts with dispensing steps, then continues with incubation/wash/etc.\n"
                "  Do NOT flag 'shake before dispensing' if the shake is part of the [5] automation flow.\n"
                "\n"
                "PLATE PREPARATION BOUNDARY CHECK (CRITICAL):\n"
                "Coating/blocking (Plate Preparation) must NEVER be included in the [5] automation!\n"
                "- If [5] includes coating (Capture Ab dispense + overnight incubation) or blocking (Block Buffer dispense + incubation),\n"
                "  FAIL.\n"
                "- The first step of [5] must be a dispense AFTER coating/blocking.\n"
                "  e.g. Assay Diluent dispense, Standard/Sample/Blank dispense, etc.\n"
                "- If the sequence has commands corresponding to coating/blocking (aspirate->dispense from a Capture Ab Reservoir,\n"
                "  aspirate->dispense from a Block Buffer Reservoir, overnight incubation), FAIL.\n"
            ),
            "schema": (
                'Return JSON ONLY: {"result":"PASS or FAIL","issues":[{"expected":"...","observed":"...","indices":[]}]}\n'
                "CRITICAL RULES:\n"
                "- ONLY include actual errors in issues. If ordering is CORRECT, do NOT add it to issues.\n"
                "- If all ordering is correct, return {\"result\":\"PASS\",\"issues\":[]}.\n"
                "- issues list must contain ONLY genuinely wrong orderings.\n"
                "- Do NOT list items you verified as correct. Correct orderings are NOT issues.\n"
                "- If your analysis concludes 'acceptable', 'correct', 'per conventions',\n"
                "  or 'not an error' — that is NOT an issue. Do NOT add it to the issues list.\n"
                "- Efficiency concerns (unnecessary TRANS round-trips) are NOT errors.\n"
                "Keep each issue SHORT. Maximum 5 issues. No prose outside JSON."
            ),
        },
    }

    def __init__(self, model: str, rules_xlsx_path: Optional[str] = None):
        self.model = model
        self._backend = self._detect_backend(model)  # "anthropic" | "openai" | "together"

        if self._backend == "anthropic":
            api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("CLAUDE_API_KEY (or ANTHROPIC_API_KEY) env var is missing.")
            self.client = Anthropic(api_key=api_key)
            available = []
            try:
                ms = self.client.models.list()
                for m in getattr(ms, "data", []) or ms:
                    mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                    if mid:
                        available.append(mid)
            except Exception:
                pass
            if available and self.model not in available:
                self.model = available[0]
            self.available_models = available
        else:
            # OpenAI / Together AI
            from openai import OpenAI as _OpenAI
            if self._backend == "together":
                api_key = os.environ.get("LLAMA_API_KEY", "")
                self.client = _OpenAI(api_key=api_key, base_url=_TOGETHER_BASE_URL)
                self.model = _LLAMA_MODEL_MAP.get(model, model)
            else:  # openai
                api_key = os.environ.get("OPENAI_API_KEY", "")
                self.client = _OpenAI(api_key=api_key)
            self.available_models = []

        self.rules_text = _read_rules_xlsx_as_text(rules_xlsx_path) if rules_xlsx_path else ""

    @staticmethod
    def _detect_backend(model: str) -> str:
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith("llama"):
            return "together"
        return "openai"  # gpt-*, o4-mini, etc.

    def _call_stage(
        self,
        stage_name: str,
        original_text: str,
        structured_text: str,
        seq_dump: str,
        attempt: int,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        wellplate_text: str = "",
    ) -> Dict[str, Any]:
        """Single-stage validation call"""
        cfg = self.STAGE_CONFIGS[stage_name]

        system = (
            "You are a strict Validation Agent for lab protocol automation.\n"
            "Return ONLY valid JSON. No prose, no markdown, no code fences.\n"
            "Keep evidence text SHORT — use sequence index numbers, not full command text.\n"
            "\n"
            "IMPORTANT: Only report ACTUAL ERRORS as issues.\n"
            "If you verify something and it is CORRECT, it is NOT an issue — do NOT include it.\n"
            "An empty issues list with result=PASS means everything checked out correctly.\n"
            "Do NOT use issues as a 'verification log'. Issues = errors ONLY.\n"
            "\n"
            "OVERINCLUSION CHECK (HIGHEST PRIORITY — BIDIRECTIONAL VERIFICATION)\n"
            "Verification must be bidirectional:\n"
            "  1) In the PDF but not in STRUCTURED -> FAIL (omission)\n"
            "  2) Not in the PDF but in STRUCTURED -> FAIL (hallucination/overinclusion)\n"
            "GPT (LLM) can invent and add steps that are not in the PDF (hallucination).\n"
            "Compare every step in STRUCTURED [5] one-to-one against the ORIGINAL_PROTOCOL PDF.\n"
            "If a step not described in the PDF is included in [5], you must report FAIL!\n"
            "e.g. extra wash not in the PDF, extra incubation not in the PDF, dispensing of a reagent not in the PDF, etc.\n"
            "\n"
            "PARAMETER PRIORITY: ORIGINAL_PROTOCOL PDF values ALWAYS override RULES defaults.\n"
            "RULES defaults (shake_speed, volume_ul, etc.) are fallbacks for when PDF is silent.\n"
            "If PDF says '500 rpm', sequence using 500 is CORRECT — do NOT flag it against default of 5.\n"
            "\n"
            "FEEDBACK QUALITY (CRITICAL for regeneration):\n"
            "When reporting issues, your feedback must be ACTIONABLE and SPECIFIC:\n"
            "- 'expected' field: state the EXACT correct value/action with FULL CONTEXT:\n"
            "  * For Reservoir mapping issues: specify exact section name (e.g. '[4] Reservoir mapping'),\n"
            "    exact item to remove (e.g. 'Reservoir 17th: Block Buffer (1% BSA in PBS)'),\n"
            "    AND where it should remain if applicable (e.g. 'keep only in the [2] user manual steps').\n"
            "  * For value issues: state the EXACT correct value (e.g. 'WAIT_MOTION 1500' not just 'longer time')\n"
            "  * Always add: 'Do not change the other sections apart from this item' to prevent over-correction.\n"
            "- 'observed' field: state ONLY the conclusion — what is wrong (e.g. 'WAIT_MOTION 60 at index 45').\n"
            "  Do NOT include thinking process ('But wait', 'Let me re-check', 'Actually...').\n"
            "- 'indices' field: provide the EXACT sequence row indices where the problem occurs\n"
            "- The regenerator (GPT) will use your feedback to fix the structured output.\n"
            "  Vague feedback = GPT cannot fix the issue and may over-correct, destroying correct content. Be precise!\n"
            "\n"
            "GLOBAL RULE — CAP_OPEN / CAP_CLOSE PARAMETERS:\n"
            "A#CAP_OPEN and A#CAP_CLOSE ALWAYS take parameters '1 9'. This is CORRECT.\n"
            "Do NOT flag 'CAP_CLOSE with parameters' or 'CAP_OPEN with parameters' as errors.\n"
            "\n"
            "GLOBAL RULE — TIP POSITIONS:\n"
            "Do NOT validate tip coordinate positions AT ALL. Tip positions are managed by the system's\n"
            "TipAllocator which shares a used-set between 4ch and 1ch. Positions will have gaps and\n"
            "appear out of order. This is CORRECT. Never flag tip position order/gaps/skips/reuse.\n"
            "\n"
            "GLOBAL RULE — 4ch/1ch DISPENSING STRUCTURE:\n"
            "In multi-channel mode, each dispensing step (e.g. 'Reservoir 3 100µL/웰 분주') is split into\n"
            "a 4ch block followed by a 1ch block in the sequence. This is NORMAL.\n"
            "Incubation (TRANS/TEMPPLATE_ON/WAIT_MOTION/TRANS) appears AFTER both 4ch and 1ch dispense blocks.\n"
            "Do NOT flag incubation position as 'interleaved' or 'between 4ch and 1ch'. The correct order is:\n"
            "  4ch dispense -> 1ch dispense -> incubation -> next step.\n"
            "This is the system's standard behavior — do NOT report it as an error.\n"
            "\n"
            "GLOBAL RULE — TYPE-BASED RESERVOIR DISPENSING:\n"
            "Reservoirs 1~N (Blank/Standard/Sample) each dispense to ONLY their designated wells, not all wells.\n"
            "Each type-based Reservoir appears as a separate dispense block in the sequence.\n"
            "The sequence will show multiple aspirate/dispense pairs from different Reservoir positions (slot 5, col 1~N)\n"
            "each targeting different subsets of wells. This is CORRECT.\n"
            "Replicates/triplicates are included automatically via well coordinates — the number of dispense commands\n"
            "per Reservoir depends on the UI well layout, NOT on the number of lines in [5].\n"
            "Do NOT flag type-based Reservoirs for 'dispensing to only a subset of wells'.\n"
            "Do NOT flag 'only 1 dispense line per Reservoir' as missing replicates.\n"
            "\n"
            "GLOBAL RULE — ROW-BASED VARIABLE VOLUME DISPENSING:\n"
            "Some protocols (Bradford, serial dilution) dispense different volumes per Row from the SAME Reservoir.\n"
            "In the sequence, this appears as multiple ASPIRATE/DISPENSE blocks from the same Reservoir column\n"
            "(slot 5, same col) each targeting a different plate Row with a different volume.\n"
            "This is CORRECT — do NOT flag as 'duplicate dispense', 'inconsistent volume', or 'repeated Reservoir'.\n"
            "Tip reuse across Rows for the same Reservoir is CORRECT (dispense = tips don't touch wells).\n"
            "The number of 4ch groups per Row depends on the plate column count:\n"
            "  - 8 columns: 2 groups (Y=1, Y=5) per Row — this is CORRECT, do NOT expect Y=9.\n"
            "  - 12 columns: 3 groups (Y=1, Y=5, Y=9) per Row.\n"
            "Do NOT assume 12 columns unless the sequence explicitly shows Y=9 dispenses.\n"
            "\n"
            "GLOBAL RULE — PLATE COORDINATE SYSTEM (CRITICAL — READ CAREFULLY):\n"
            "ASPIRATE_FROM_PLATE and DISPENSE_INTO_PLATE parameters: (slot, bioforge_col, bioforge_row, volume, speed).\n"
            "- bioforge_col = plate COLUMN index (1~12 for 96-well plate, maps to columns 1-12).\n"
            "- bioforge_row = plate ROW index (1~8, maps to rows A-H).\n"
            "The 2nd parameter is COLUMN, NOT row. A 96-well plate has 12 columns (1~12) and 8 rows (1~8).\n"
            "Values of 9, 10, 11, 12 in the 2nd parameter are VALID column indices, NOT row indices.\n"
            "Do NOT flag bioforge_col values of 9-12 as 'exceeding plate row limit'. They are columns.\n"
            "The 3rd parameter (bioforge_row) ranges 1~8 only. Flag errors only if bioforge_row > 8.\n"
            "\n"
            "GLOBAL RULE — INSTALL_PIPETTE COORDINATES ARE TIP POSITIONS, NOT PLATE POSITIONS:\n"
            "A#ADP_INSTALL_PIPETTE parameters are (tip_slot, tip_row, tip_col) — these are coordinates\n"
            "in the TIP RACK (slot 4), NOT on the sample plate (slot 1).\n"
            "Tip rack rows can go up to 12 or higher. Values like '4 9 1', '4 10 1', '4 11 1', '4 12 1'\n"
            "mean tip_slot=4, tip_row=9/10/11/12, tip_col=1. These are VALID tip positions.\n"
            "Do NOT confuse INSTALL_PIPETTE coordinates with DISPENSE_INTO_PLATE coordinates.\n"
            "Only ASPIRATE_FROM_PLATE and DISPENSE_INTO_PLATE have plate coordinates.\n"
            "Do NOT flag INSTALL_PIPETTE rows >= 9 as 'exceeding plate row limit'.\n"
            "\n"
            "GLOBAL RULE — TIP CONTAMINATION PREVENTION:\n"
            "Tips that TOUCH wells (wash aspiration, solution removal) must be changed per group (4ch) or per well (1ch).\n"
            "- WASH: round-robin with 4ch/1ch alternation per round.\n"
            "  [each round]: 4ch round (each group: INSTALL->work->EJECT) -> 1ch round (each well: INSTALL->work->EJECT).\n"
            "  i.e. 4ch wash round 1 -> 1ch wash round 1 -> 4ch wash round 2 -> 1ch wash round 2 -> ...\n"
            "- REMOVE: per group/well — [each group/well]: INSTALL->well_asp->waste->EJECT.\n"
            "- DISPENSE: tips do NOT touch wells -> single INSTALL, all groups/wells, single EJECT.\n"
            "This means wash/remove blocks will have MANY INSTALL/EJECT pairs (one per group per round for wash,\n"
            "one per group for remove). This is CORRECT and required for contamination prevention.\n"
            "Do NOT flag frequent INSTALL/EJECT in wash/remove blocks as errors or 'unnecessary'.\n"
            "\n"
            "GLOBAL RULE — COMMON REAGENT RESERVOIR SWAP TOLERANCE (CRITICAL):\n"
            "A change in the Reservoir-number order of common reagents (Wash Buffer, Assay Diluent, etc.) is not an error.\n"
            "Example: if per the PDF Wash Buffer=Reservoir 14, Assay Diluent=Reservoir 15, but\n"
            "    STRUCTURED assigns Wash Buffer=Reservoir 15, Assay Diluent=Reservoir 14 -> PASS.\n"
            "When only the Reservoir numbers are swapped between common reagents, it is fine as long as the reagent contents and volumes match.\n"
            "The corresponding Reservoir column numbers appearing differently in the SEQUENCE is naturally also fine.\n"
            "Do NOT report this swap as an error!\n"
            "\n"
            "GLOBAL RULE — SHAKING PARAMETER TOLERANCE:\n"
            "For 'short shaking (mixing)' or 'gently tap', the system automatically applies the defaults (200 rpm, 10 s).\n"
            "Therefore, even if STRUCTURED has no rpm/time, 200 rpm / 10 s appearing in the SEQUENCE is fine.\n"
            "Conversely, STRUCTURED stating 'Shaker 200 rpm, 10 s (mixing)' is also fine.\n"
            "Do NOT report rpm/time differences in shaking as errors!\n"
            "\n"
            "GLOBAL RULE — RESERVOIR COUNT HAS NO LIMIT (CRITICAL):\n"
            "Deck 5 can hold ANY number of Reservoirs. There is NO upper limit (not 12, not any number).\n"
            "Reservoir columns 13, 14, 15, 16, 17, 18, 19, 20+ are ALL VALID.\n"
            "Do NOT flag Reservoir numbers above 12 as 'exceeding deck capacity' or 'exceeding slot limit'.\n"
            "Do NOT flag Reservoir numbers above 12 as errors for ANY reason related to count or capacity.\n"
            "The Deck_Map may historically show '1..12' but this restriction has been REMOVED.\n"
            "Even if RULES/DECK_MAP text says '1..12', IGNORE that limit — it is outdated.\n"
            "If a protocol needs 15, 17, or 20 Reservoirs, that is perfectly valid.\n"
            "Only flag Reservoir issues if the MAPPING is wrong (wrong reagent assigned), NOT the number.\n"
            "\n"
            "GLOBAL RULE — PLATE PREPARATION BOUNDARY (CRITICAL — HIGHEST PRIORITY):\n"
            "Coating/blocking (Plate Preparation) must NEVER be included in the [5] automation!\n"
            "- Capture Antibody coating (dispense + overnight/multi-hour incubation) exists only in the [2] user manual steps.\n"
            "- Blocking (Block Buffer dispense + incubation) also exists only in [2].\n"
            "- The wash after coating and the wash after blocking are also included in [2].\n"
            "- If STRUCTURED [5] has any coating/blocking step -> you must report FAIL!\n"
            "- If the [4] Reservoir mapping has Capture Ab or Block Buffer -> you must report FAIL!\n"
            "- If the SEQUENCE has coating/blocking commands -> you must report FAIL!\n"
            "- Always apply regardless of kit type (DuoSet, Development Kit, KIMM, etc.).\n"
            "- Starting point of [5]: the first dispense after coating/blocking is complete (Standard/Sample/Assay Diluent, etc.).\n"
            "- This rule is for overinclusion detection.\n"
            "  Check not only 'is anything missing' but also 'is anything present that should not be'!\n"
        )

        plate_section = (
            f"[PLATE_LAYOUT]\n{wellplate_text}\n\n"
            if wellplate_text and wellplate_text.strip()
            else ""
        )
        user = (
            f"[ATTEMPT] {attempt}\n\n"
            f"[RULES]\n{self.rules_text}\n\n"
            f"[ORIGINAL_PROTOCOL]\n{original_text}\n\n"
            f"[STRUCTURED_OUTPUT]\n{structured_text}\n\n"
            f"[COMMAND_SEQUENCE]\n{seq_dump}\n\n"
            f"{plate_section}"
            f"[TASK]\n{cfg['task']}\n\n"
            f"[OUTPUT]\n{cfg['schema']}"
        )

        last_err = None
        for retry in range(3):
            try:
                if self._backend == "anthropic":
                    resp = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    text = resp.content[0].text
                else:
                    # OpenAI / Together AI: place the system message first in messages
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    text = resp.choices[0].message.content
                parsed = _extract_json(text)
                # False-positive filtering: remove normal items such as "correct"/"no error"
                raw_issues = parsed.get("issues", []) or []
                filtered = _filter_false_positives(raw_issues)
                parsed["issues"] = filtered
                # Keep the result judged by Claude as-is (no automatic correction)
                return parsed
            except json.JSONDecodeError as e:
                last_err = e
                time.sleep(1.0)
            except Exception as e:
                last_err = e
                time.sleep(1.5)

        # Default on failure
        return {"result": "UNKNOWN", "issues": [], "_error": str(last_err)}

    def validate(
        self,
        original_text: str,
        structured_text: str,
        seq_df: pd.DataFrame,
        attempt: int,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        wellplate_text: str = "",
    ) -> Dict[str, Any]:
        """Call the 3 stages individually, then aggregate"""
        seq_dump = _sequence_df_to_dump(seq_df)

        # -- Code-level preparation boundary validation (Claude backstop) --
        prep_boundary_issues = _check_preparation_boundary(structured_text)

        # -- Code-level sequence validation: volume, Stop Solution, etc. (Claude backstop) --
        code_level_issues = _code_level_sequence_checks(structured_text, seq_df)

        stages = {}
        all_issues = []

        for stage_name in ["completeness", "parameter_accuracy", "execution_order"]:
            result = self._call_stage(
                stage_name, original_text, structured_text,
                seq_dump, attempt, max_tokens, temperature,
                wellplate_text=wellplate_text,
            )

            # -- Code-level sequence validation: remove Claude hallucination false positives --
            stage_issues = result.get("issues", [])
            stage_issues = _filter_sequence_false_positives(stage_issues, seq_df)

            # -- Merge preparation boundary issues into the completeness stage --
            if stage_name == "completeness" and prep_boundary_issues:
                stage_issues.extend(prep_boundary_issues)
                result["result"] = "FAIL"  # Force FAIL on a coating/blocking boundary violation

            # -- Merge code-level sequence issues into the parameter_accuracy stage --
            if stage_name == "parameter_accuracy" and code_level_issues:
                stage_issues.extend(code_level_issues)
                result["result"] = "FAIL"

            result["issues"] = stage_issues
            # Ghost FAIL correction: if result is FAIL but there are 0 issues, set to PASS (validator FAILed without grounds)
            if not stage_issues and result.get("result", "").upper() in ("FAIL", "UNKNOWN"):
                result["result"] = "PASS"

            stages[stage_name] = {
                "result": result.get("result", "UNKNOWN"),
                "issues": stage_issues,
            }
            if result.get("_error"):
                stages[stage_name]["_error"] = result["_error"]

            all_issues.extend(stage_issues)

        # Overall judgment
        # UNKNOWN is kept as-is (no automatic correction)

        stage_results = [s["result"] for s in stages.values()]
        if "FAIL" in stage_results:
            overall = "FAIL"
        elif "UNKNOWN" in stage_results:
            overall = "FAIL"
        else:
            overall = "PASS"

        # must_fix: top 3 by severity after deduplication
        must_fix = _deduplicate_and_prioritize(all_issues)[:3] if overall == "FAIL" else []

        # feedback_to_regenerator
        if overall == "FAIL":
            fail_stages = [name for name, s in stages.items() if s["result"] == "FAIL"]
            has_completeness = "completeness" in fail_stages
            has_parameter = "parameter_accuracy" in fail_stages
            has_execution = "execution_order" in fail_stages

            if has_completeness and has_parameter:
                target = "BOTH"
            elif has_completeness:
                target = "STRUCTURING"
            elif has_parameter:
                target = "MAPPING"
            elif has_execution:
                target = "STRUCTURING"  # only execution_order failed -> resolved by fixing structured
            else:
                target = "BOTH"

            instructions_parts = []
            for issue in must_fix:
                exp = issue.get("expected", "")
                obs = issue.get("observed", "")
                indices = issue.get("indices", [])
                if exp or obs:
                    idx_str = f" (sequence indices: {indices})" if indices else ""
                    instructions_parts.append(f"- Problem: {obs}{idx_str}\n  Fix: {exp}")
            instructions = "\n".join(instructions_parts) if instructions_parts else "Review failed stages."

            # State the names of the failed stages
            fail_stage_names = [name for name, s in stages.items() if s["result"] == "FAIL"]
            stage_info = f"[Failed stages: {', '.join(fail_stage_names)}]\n"
            preservation_note = (
                "\n\n[Regeneration scope limit — must be followed]\n"
                "- Make minimal changes to only the items flagged above. Keep everything else identical to the previous version.\n"
                "- An instruction to remove a specific reagent from the Reservoir mapping -> remove only that one line.\n"
                "  Do not change that reagent's dispense/aspirate sequence steps or other reagents' steps.\n"
                "- The total line count must not change significantly before and after the fix (remove only the flagged items)."
            )
            instructions = stage_info + instructions + preservation_note

            feedback = {"target": target, "instructions": instructions}
        else:
            feedback = {"target": "", "instructions": ""}

        return {
            "overall": overall,
            "stages": stages,
            "must_fix": must_fix,
            "feedback_to_regenerator": feedback,
        }
