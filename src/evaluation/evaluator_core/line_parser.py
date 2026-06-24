"""
line_parser.py — Normalize Structured Protocol section [5] lines into Action tuples.
Both ground-truth and generated files are parsed with the same parser to produce a
comparable representation.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Action:
    """A normalized action unit"""
    type: str                       # incubation|solution-removal|wash|dispense|shaking|readout|stop-dispense
    volume: Optional[int] = None    # µL
    time_sec: Optional[int] = None  # seconds
    temperature: Optional[int] = None  # °C
    wash_count: Optional[int] = None   # count
    reservoir: Optional[int] = None    # reservoir number
    rpm: Optional[int] = None          # shaking RPM
    wavelength: Optional[int] = None   # nm
    raw_line: str = ""                 # original text (for debugging)

    def match_key(self) -> Tuple:
        """LCS matching key — type only. Reservoir excluded for all types
        because Wash Buffer position change can cascade-shift all reservoir numbers.
        Reservoir differences are caught as parameter errors instead."""
        return (self.type,)

    def param_dict(self) -> dict:
        """Return comparable parameters (excluding None).
        Reservoir excluded for all types — Wash Buffer position change
        can cascade-shift all reservoir numbers. Reservoir mapping is
        evaluated separately by reservoir_evaluator_llm."""
        d = {}
        if self.volume is not None:
            d["volume"] = self.volume
        if self.time_sec is not None:
            d["time_sec"] = self.time_sec
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.wash_count is not None:
            d["wash_count"] = self.wash_count
        # reservoir excluded — evaluated separately in reservoir mapping
        if self.rpm is not None:
            d["rpm"] = self.rpm
        if self.wavelength is not None:
            d["wavelength"] = self.wavelength
        return d

    def short_desc(self) -> str:
        """A short, human-readable description"""
        parts = [self.type]
        if self.reservoir is not None:
            parts.append(f"R{self.reservoir}")
        if self.volume is not None:
            parts.append(f"{self.volume}µL")
        if self.time_sec is not None:
            parts.append(f"{self.time_sec}s")
        if self.wash_count is not None:
            parts.append(f"{self.wash_count}회")
        if self.rpm is not None:
            parts.append(f"{self.rpm}rpm")
        if self.wavelength is not None:
            parts.append(f"{self.wavelength}nm")
        return " ".join(parts)


def _parse_time_to_sec(text: str) -> Optional[int]:
    """Convert a time expression to seconds"""
    # "2시간" (2 hours) -> 7200
    m = re.search(r"(\d+(?:\.\d+)?)\s*시간", text)
    if m:
        return int(float(m.group(1)) * 3600)

    # "30분" (30 minutes) -> 1800
    m = re.search(r"(\d+(?:\.\d+)?)\s*분", text)
    if m:
        return int(float(m.group(1)) * 60)

    # "1800초" (1800 seconds) -> 1800
    m = re.search(r"(\d+)\s*초", text)
    if m:
        return int(m.group(1))

    # "1800sec" / "1800s"
    m = re.search(r"(\d+)\s*s(?:ec)?(?:\b|$)", text, re.I)
    if m:
        return int(m.group(1))

    return None


def _parse_volume(text: str) -> Optional[int]:
    """Extract the volume (µL)"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[µμuU]?[Ll]/웰", text)
    if m:
        return int(float(m.group(1)))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[µμuU]?[Ll]", text)
    if m:
        return int(float(m.group(1)))
    return None


def _parse_reservoir_no(text: str) -> Optional[int]:
    """Extract the reservoir number"""
    m = re.search(r"[Rr]eservoir\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def _parse_wash_count(text: str) -> Optional[int]:
    """Extract the wash count"""
    m = re.search(r"(\d+)\s*회", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*times", text, re.I)
    if m:
        return int(m.group(1))
    return None


def _parse_wavelength(text: str) -> Optional[int]:
    """Extract the wavelength (nm)"""
    m = re.search(r"(\d+)\s*nm", text, re.I)
    if m:
        return int(m.group(1))
    return None


def _parse_rpm(text: str) -> Optional[int]:
    """Extract the RPM"""
    m = re.search(r"(\d+)\s*rpm", text, re.I)
    if m:
        return int(m.group(1))
    return None


def parse_instrument_line(line: str) -> Optional[Action]:
    """
    Convert a single line of automation-execution section [5] into an Action.
    Returns None for lines that cannot be parsed or should be ignored (section headers, etc.).
    """
    line = line.strip()
    if not line or line.startswith("[5]") or line.startswith("[6]"):
        return None

    # Remove bullet
    line_clean = re.sub(r"^[-•*]\s*", "", line).strip()
    if not line_clean:
        return None

    # ── Incubation ──
    if re.search(r"배양|incubat", line_clean, re.I):
        # This is unlikely to be the incubation right after stop-solution dispensing (that is a separate line)
        time_sec = _parse_time_to_sec(line_clean)
        temp = 20  # default room temperature
        m_temp = re.search(r"(\d+)\s*[°℃]", line_clean)
        if m_temp:
            temp = int(m_temp.group(1))
        return Action(type="배양", time_sec=time_sec, temperature=temp, raw_line=line)

    # ── Solution removal ──
    if re.search(r"용액\s*제거|aspirat", line_clean, re.I):
        vol = _parse_volume(line_clean)
        return Action(type="용액제거", volume=vol, raw_line=line)

    # ── Wash ──
    if re.search(r"세척|[Ww]ash", line_clean):
        reservoir = _parse_reservoir_no(line_clean)
        count = _parse_wash_count(line_clean)
        vol = _parse_volume(line_clean)
        return Action(type="세척", reservoir=reservoir, wash_count=count, volume=vol, raw_line=line)

    # ── Dispense (Reservoir) ──
    if re.search(r"[Rr]eservoir.*분주|분주.*[Rr]eservoir", line_clean):
        reservoir = _parse_reservoir_no(line_clean)
        vol = _parse_volume(line_clean)
        return Action(type="분주", reservoir=reservoir, volume=vol, raw_line=line)

    # ── shaking / tap ──
    if re.search(r"shaking|shake|두드|tap", line_clean, re.I):
        time_sec = _parse_time_to_sec(line_clean)
        rpm = _parse_rpm(line_clean)
        return Action(type="shaking", time_sec=time_sec, rpm=rpm, raw_line=line)

    # ── O.D. readout ──
    if re.search(r"O\.?D\.?\s*판독|판독|absorbance|read", line_clean, re.I):
        wl = _parse_wavelength(line_clean)
        return Action(type="판독", wavelength=wl, raw_line=line)

    # ── Other dispense (dispense without a reservoir) ──
    if re.search(r"분주", line_clean):
        vol = _parse_volume(line_clean)
        reservoir = _parse_reservoir_no(line_clean)
        return Action(type="분주", reservoir=reservoir, volume=vol, raw_line=line)

    # Parsing failed — return anyway (unknown)
    return Action(type="unknown", raw_line=line)


def parse_section5(text: str) -> List[Action]:
    """Parse the entire automation-execution section [5] and return a list of Actions"""
    actions = []
    in_section5 = False

    for line in text.splitlines():
        stripped = line.strip()

        # Detect the start of [5]
        if re.match(r"\[5\]", stripped):
            in_section5 = True
            continue

        # Start of [6] -> end of [5]
        if re.match(r"\[6\]", stripped):
            break

        if not in_section5:
            continue

        action = parse_instrument_line(stripped)
        if action and action.type != "unknown":
            actions.append(action)

    return actions


def parse_section6(text: str) -> List[Action]:
    """Parse the readout/data-processing section [6]"""
    actions = []
    in_section6 = False

    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"\[6\]", stripped):
            in_section6 = True
            continue
        if in_section6 and re.match(r"\[", stripped):
            break
        if not in_section6:
            continue

        action = parse_instrument_line(stripped)
        if action and action.type != "unknown":
            actions.append(action)

    return actions


@dataclass
class ReservoirEntry:
    """A reservoir mapping entry"""
    number: int
    reagent_raw: str
    reagent_normalized: str


def _normalize_reagent(name: str) -> str:
    """Normalize a reagent name — lowercase, clean whitespace/special characters, unify abbreviations"""
    s = name.lower().strip()
    # Remove concentration/dilution information
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\d+[\.\d]*\s*[µμu]?g/m[lL]", "", s)
    s = re.sub(r"\d+[\.\d]*\s*[µμu]?[lL]/웰.*", "", s)
    s = re.sub(r"\d+[\.\d]*\s*[µμu]?[lL]", "", s)
    # Unify abbreviations
    s = re.sub(r"\bab\b", "antibody", s)
    s = re.sub(r"\bdet\b", "detection", s)
    s = re.sub(r"\bsa-hrp\b", "streptavidin-hrp", s)
    s = re.sub(r"\bhrp\b", "hrp", s)
    # Clean up whitespace
    s = re.sub(r"[,;:·•\-–—]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Keep only the general keywords
    return s


def parse_reservoir_mapping(text: str) -> List[ReservoirEntry]:
    """Parse the reservoir mapping section [4]"""
    entries = []
    in_section4 = False

    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"\[4\]", stripped):
            in_section4 = True
            continue
        if in_section4 and re.match(r"\[5\]", stripped):
            break
        if in_section4 and stripped.startswith("</"):
            break
        if not in_section4:
            continue

        # Format 1: "Reservoir N번째: reagent-name ..." or "Reservoir N: reagent-name ..."
        m = re.match(r"[-•*]?\s*[Rr]eservoir\s*(\d+)\s*(?:번째)?\s*[:：]\s*(.+)", stripped)
        if m:
            num = int(m.group(1))
            raw = m.group(2).strip()
            reagent_part = re.split(r"\s*\(", raw, maxsplit=1)[0].strip()
            entries.append(ReservoirEntry(
                number=num,
                reagent_raw=raw,
                reagent_normalized=_normalize_reagent(reagent_part),
            ))
            continue

        # Format 2: "Reservoir N (reagent-name, ...)" — reagent name inside parentheses
        m = re.match(r"[-•*]?\s*[Rr]eservoir\s*(\d+)\s*\((.+)\)\s*$", stripped)
        if not m:
            # When there is extra text after the parentheses: "Reservoir 1 (Wash Buffer) 300µL/웰 세척, 4회"
            m = re.match(r"[-•*]?\s*[Rr]eservoir\s*(\d+)\s*\((.+?)\)", stripped)
        if m:
            num = int(m.group(1))
            raw = m.group(2).strip()
            # Inside the parentheses: before the comma = reagent name, after = additional information
            reagent_part = raw.split(",")[0].strip()
            entries.append(ReservoirEntry(
                number=num,
                reagent_raw=raw,
                reagent_normalized=_normalize_reagent(reagent_part),
            ))

    return entries


def parse_natural_sections(text: str) -> Dict[str, List[str]]:
    """
    Split sections [1]–[4] inside the <NATURAL> tag into lists of lines.
    Returns: {"1": [line, ...], "2": [...], "3": [...], "4": [...]}
    """
    sections: Dict[str, List[str]] = {}
    current_section = None

    for line in text.splitlines():
        stripped = line.strip()
        # Detect section headers: [1], [2], [3], [4]
        m = re.match(r"\[(\d)\]", stripped)
        if m:
            sec_num = m.group(1)
            if sec_num in ("1", "2", "3", "4"):
                current_section = sec_num
                sections[current_section] = []
                continue
            else:
                # [5] or higher -> end of the NATURAL section
                break

        if current_section is None:
            continue

        # Ignore blank lines and tag lines
        if not stripped or stripped.startswith("<") or stripped.startswith("</"):
            continue

        # Ignore standalone "미기재" (not-specified) lines (no meaningful comparison)
        if stripped == "미기재":
            continue

        sections.setdefault(current_section, []).append(stripped)

    return sections


def extract_numeric_params(line: str) -> List[Tuple[str, str]]:
    """
    Extract (value + unit) pairs from a free-form line.
    Examples: "500 µL씩" -> [("500", "µL")], "0.25 µg/mL" -> [("0.25", "µg/mL")]
    """
    results = []
    # Volume: 300 µL, 100 µL/웰, 500µL
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[µμuU]?[Ll](?:/웰)?", line):
        results.append((m.group(1), "µL"))
    # Concentration: 0.25 µg/mL, 10,000 pg/mL
    for m in re.finditer(r"([\d,]+(?:\.\d+)?)\s*(?:pg|ng|[µμu]g)/m[lL]", line):
        results.append((m.group(1).replace(",", ""), "conc"))
    # Count: 4회
    for m in re.finditer(r"(\d+)\s*회", line):
        results.append((m.group(1), "회"))
    # Percentage: 0.05%, 1%
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", line):
        results.append((m.group(1), "%"))
    # Time: 30분, 2시간
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*분", line):
        results.append((m.group(1), "분"))
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*시간", line):
        results.append((m.group(1), "시간"))
    # Temperature: 37°C
    for m in re.finditer(r"(\d+)\s*[°℃]", line):
        results.append((m.group(1), "°C"))
    # Magnification: 100배, 25X
    for m in re.finditer(r"(\d+)\s*배", line):
        results.append((m.group(1), "배"))
    for m in re.finditer(r"(\d+)\s*[xX](?:\s|$)", line):
        results.append((m.group(1), "X"))
    return results


# ── Filtering: kit reconstitution/storage vs. user-performed parameters ──

# Keywords identifying kit reconstitution/storage lines (exclude certain units on lines containing these keywords)
_KIT_RECON_KEYWORDS = [
    "재구성", "재용해", "용해", "reconstitut",
    "소분", "aliquot", "분주 보관", "분주보관",
    "원액 준비", "원액준비",            # "Streptavidin-HRP stock preparation"
    "작업용액", "working solution",     # "Detection Ab working solution: with Diluent ..."
    "1:2000", "1:1000", "1:500",       # "dilute Avidin-HRP 1:2000"
    "작업 농도로 희석",                  # "dilute Capture Ab to working concentration in PBS"
]
_KIT_STORAGE_KEYWORDS = [
    "보관", "동결", "냉동", "냉장", "암소", "암상", "저장",
    "storage", "store", "freeze",
]
_KIT_BUFFER_DEF_KEYWORDS = [
    # Buffer definition lines of the form "[material]:"
    "wash buffer:", "block buffer:", "blocking buffer:",
    "diluent:", "assay diluent:",
    "wash buffer :", "block buffer :", "diluent :",
]
_KIT_PBS_KEYWORDS = [
    "10x", "10X", "1x pbs", "1X PBS", "1xpbs", "1xPBS",
]
_KIT_SUBSTRATE_MIX_KEYWORDS = [
    "color reagent a", "color reagent b",
    "substrate solution",
    "1:1로 혼합", "1:1 혼합",
]
_KIT_STD_CASCADE_KEYWORDS = [
    "→", "단계 희석", "단계적으로 희석", "단계희석", "serial dilut",
    "순차적으로 옮", "다음 튜브",
    "연속 희석", "표준 희석", "희석 시리즈", "희석시리즈",
    "표준곡선 준비", "표준 곡선 준비",
    "계단 희석", "표준곡선",             # "2-fold stepwise dilution into a 7-point standard curve"
]

# Identify Wash Buffer preparation lines (e.g., 25X concentrate -> 1X)
_KIT_WASH_BUFFER_PREP_KEYWORDS = [
    "wash buffer",             # "Wash Buffer 25X Concentrate 20 mL ..."
    "calibrator diluent",      # "Calibrator Diluent RD5P (1:5 dilution): ..."
]

# Sample dilution (recommended) lines — recipe defined by the kit manual
_KIT_SAMPLE_DILUTION_KEYWORDS = [
    "시료 희석", "샘플 희석", "sample dilut",
]


def _is_kit_recon_line(line: str) -> bool:
    """Determine whether a line is a kit reconstitution/storage/buffer-definition/wash-buffer-preparation line"""
    lower = line.lower()

    # Buffer definition lines (a definition statement starting with ": ")
    # "- Wash Buffer: prepared with 0.05% Tween-20 in PBS"
    for kw in _KIT_BUFFER_DEF_KEYWORDS:
        if kw in lower:
            return True

    # Reconstitution/storage keywords
    for kw in _KIT_RECON_KEYWORDS:
        if kw in lower:
            return True

    # Storage keywords
    for kw in _KIT_STORAGE_KEYWORDS:
        if kw in lower:
            return True

    # PBS preparation (10x->1x)
    for kw in _KIT_PBS_KEYWORDS:
        if kw in line:  # case-sensitive
            return True

    # Substrate mixing
    for kw in _KIT_SUBSTRATE_MIX_KEYWORDS:
        if kw in lower:
            return True

    # Wash Buffer preparation / Calibrator Diluent preparation
    for kw in _KIT_WASH_BUFFER_PREP_KEYWORDS:
        if kw in lower:
            # Exclude execution lines such as "wash with wash buffer"
            # Preparation lines usually appear in [1] and include volume/dilution-factor information
            if "세척" not in lower and "웰" not in lower:
                return True

    # Sample dilution (recommended) lines
    for kw in _KIT_SAMPLE_DILUTION_KEYWORDS:
        if kw in lower:
            return True

    return False


def _is_std_cascade_line(line: str) -> bool:
    """Determine whether a line is a standard cascade (serial dilution) line"""
    lower = line.lower()
    for kw in _KIT_STD_CASCADE_KEYWORDS:
        if kw in lower:
            return True
    return False


def extract_user_action_params(line: str) -> List[Tuple[str, str]]:
    """
    From the natural-language sections [1]–[3], extract only the parameters the user
    **must** know.

    Sections [1]–[3] of the ELISA protocol are all manual user-performed steps. Among
    the information written here, some can be found in the kit manual (the preparation
    recipe), while other parts must be conveyed accurately by the LLM.

    ■ Must be present (keep):
       - µL  : dispense volume (e.g., dispense 100 µL/well)
       - 회  : wash count (e.g., 4 times)
       - 분/시간 : incubation time (e.g., 60 minutes, overnight)

    ■ Acceptable to omit (remove):
       - conc (pg/mL, µg/mL): concentration — manual recipe
       - %   : buffer composition (0.05% Tween-20) — manual recipe
       - X   : magnification/concentration factor (25X, 1X) — manual recipe
       - 배  : dilution factor (2x, 100x) — manual recipe
       - °C  : temperature — stated in the manual

    ■ Special cases:
       - cascade (standard dilution) line -> remove entirely (manual recipe)
       - kit reconstitution/storage/buffer line -> remove entirely

    Returns: [(value_str, unit_str), ...]
    """
    all_params = extract_numeric_params(line)

    if not all_params:
        return []

    # ── Standard cascade line (check first) ──
    # The entire cascade is part of the kit manual recipe -> remove all
    if _is_std_cascade_line(line):
        return []

    # ── Kit reconstitution/storage/buffer-definition line ──
    # Remove all (including time — "mix for 15 minutes" is also a manual recipe)
    if _is_kit_recon_line(line):
        return []

    # ── General line: keep only user-performed parameters (µL, 회, 분, 시간) ──
    # conc, %, X, 배, °C, etc. are information stated in the manual, so remove them
    kept = []
    for val, unit in all_params:
        if unit in ("µL", "회", "분", "시간"):
            kept.append((val, unit))
    return kept


def extract_tag_content(text: str, tag: str) -> str:
    """Extract the content of an XML-like tag"""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    return m.group(1).strip() if m else ""


def extract_instrument_text(full_text: str) -> str:
    """Extract the content of the <INSTRUMENT> tag"""
    return extract_tag_content(full_text, "INSTRUMENT")


def extract_natural_text(full_text: str) -> str:
    """Extract the content of the <NATURAL> tag"""
    return extract_tag_content(full_text, "NATURAL")
