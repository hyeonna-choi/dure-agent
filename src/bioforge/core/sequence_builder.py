# bioforge/core/sequence_builder.py

import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from .exceptions import ImpossibleActionDetector


class TipAllocator:
    """Tip coordinate allocator for 4-channel / 1-channel pipettes.

    Coordinate: (X, Y)  X = row (top to bottom, 1-12)  Y = column (right to left, 1-8)
    When Y is exhausted, X is incremented.

    4ch: consumes 4 consecutive slots (Y, Y+1, Y+2, Y+3). The INSTALL parameter is the
         starting Y. If the current X row has no 4 consecutive slots, move to the next X row.
    1ch: consumes a single empty slot, searched in order.
    """

    MAX_X = 12
    MAX_Y = 8

    def __init__(self, tip_slot: int):
        self.tip_slot = tip_slot
        self.used: set = set()        # record of used (x, y)

    def _find_4ch(self) -> Tuple[int, int, int] | None:
        """Find an empty 4-channel slot (returns None if none available)."""
        for x in range(1, self.MAX_X + 1):
            for y in (1, 5):                             # 4ch alignment: only 1 or 5
                if all((x, y + d) not in self.used for d in range(4)):
                    for d in range(4):
                        self.used.add((x, y + d))
                    return (self.tip_slot, x, y)
        return None

    def _find_1ch(self) -> Tuple[int, int, int] | None:
        """Find an empty 1-channel slot (returns None if none available)."""
        for x in range(1, self.MAX_X + 1):
            for y in range(1, self.MAX_Y + 1):
                if (x, y) not in self.used:
                    self.used.add((x, y))
                    return (self.tip_slot, x, y)
        return None

    def next_4ch(self) -> Tuple[int, int, int]:
        """4-channel: allocate 4 consecutive slots. When exhausted, clear `used` and reallocate from (1,1)."""
        result = self._find_4ch()
        if result is not None:
            return result
        # Exhausted -> reset and restart (operator physically replaces the tips)
        self.used.clear()
        return self._find_4ch()

    def next_1ch(self) -> Tuple[int, int, int]:
        """1-channel: allocate a single empty slot. When exhausted, clear `used` and reallocate from (1,1)."""
        result = self._find_1ch()
        if result is not None:
            return result
        # Exhausted -> reset and restart (operator physically replaces the tips)
        self.used.clear()
        return self._find_1ch()

    def next_install_param(self) -> Tuple[int, int, int]:
        """Backward compatibility: defaults to 4-channel behavior when no channel is specified."""
        return self.next_4ch()

def load_all_sheets(xlsx_path: Path) -> Dict[str, pd.DataFrame]:
    """Load all sheets from the Excel file."""
    xls = pd.ExcelFile(xlsx_path)
    return {name: xls.parse(name) for name in xls.sheet_names}


def get_allowed_commands(sheets: Dict[str, pd.DataFrame], command_whitelist: List[str]) -> List[str]:
    """Extract the list of allowed commands from the Excel file."""
    cmd_list: List[str] = []
    for _, df in sheets.items():
        norm_cols = {c: re.sub(r"\s+", "", str(c)).lower() for c in df.columns}
        cmd_col = None
        for c, n in norm_cols.items():
            if n == "command":
                cmd_col = c
                break
        if cmd_col is None:
            continue
        cmds = (
            df[cmd_col]
            .dropna()
            .astype(str)
            .map(lambda s: s.strip())
            .tolist()
        )
        cmd_list += cmds
    return sorted(set(cmd_list).intersection(command_whitelist))


def infer_deck_slots(sheets: Dict[str, pd.DataFrame], default_slots: Dict[str, int]) -> Dict[str, int]:
    """Extract the deck slot mapping."""
    return dict(default_slots)


def p(*xs) -> str:
    """Parameter formatting utility."""
    return " ".join(str(x) for x in xs)


def parse_reservoir_col(line: str, default_col: int = 1) -> int:
    """Parse the reservoir column - no limit (sum of type-specific and shared reagents)."""
    m = re.search(r"reservoir\s*(\d+)", line, flags=re.I)
    if m:
        return max(1, int(m.group(1)))

    return default_col


def extract_int(pattern: str, s: str, default: int = 0) -> int:
    """Extract an integer using a regular expression."""
    m = re.search(pattern, s, flags=re.I)
    return int(m.group(1)) if m else default


def extract_volume_ul(line: str, default_ul: int) -> int:
    """Extract the volume (uL)."""
    m = re.search(r"(\d+)\s*(?:u?µ?l)\b", line, flags=re.I)
    return int(m.group(1)) if m else default_ul


def extract_minutes(line: str, default_min: int) -> int:
    """Extract the time (minutes)."""
    # overnight / o/n -> default of 16 hours (960 minutes)
    if re.search(r"overnight|o\s*/\s*n", line, flags=re.I):
        return 960
    m1 = re.search(r"(\d+)\s*min", line, flags=re.I)
    if m1: return int(m1.group(1))
    m2 = re.search(r"(\d+)\s*분", line, flags=re.I)
    if m2: return int(m2.group(1))
    m3 = re.search(r"(\d+)\s*시간", line, flags=re.I)
    if m3: return int(m3.group(1)) * 60
    return default_min


def extract_rpm(line: str, default_rpm: int) -> int:
    """Extract the RPM."""
    # For the "500 +/- 50 rpm" form, prefer extracting the first number
    m = re.search(r"(\d+)\s*[±]\s*\d+\s*rpm", line, flags=re.I)
    if m: return int(m.group(1))
    # Generic numeric rpm form
    m = re.search(r"(\d+)\s*rpm", line, flags=re.I)
    return int(m.group(1)) if m else default_rpm


def cmd(handler: int, command: str, params: str) -> Dict[str, str]:
    """Create a sequence command."""
    return {"Handler": str(handler), "Command": command, "Input Parameters": params}


def add(out: List[Dict[str, str]], allowed: List[str], h: int, c: str, p: str):
    """Append to the list only commands that are allowed."""
    if c in allowed:
        out.append(cmd(h, c, p))


def build_wash_block(allowed, slots, tipper, repeats, volume_ul, asp_speed, disp_speed, reservoir_col: int):
    """N-time wash block: install tip once -> (aspirate -> dispense -> remove -> discard) x N -> eject tip once."""
    out: List[Dict[str, str]] = []
    ts, tc, ti = tipper.next_install_param()
    add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
    for _ in range(max(1, repeats)):
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(slots["reservoir_slot"], reservoir_col, 1, volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["plate_slot"], 1, 1, volume_ul, disp_speed))
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(slots["plate_slot"], 1, 1, volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["waste_slot"], 1, 4, volume_ul, 120))
    add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))
    return out

def build_wash_block_4ch_single(allowed, slots, tipper, coord_groups, volume_ul, asp_speed, disp_speed, reservoir_col: int):
    """4-channel wash, one round: replace tip for each group.

    Structure: [each group]: SELECT_4CH -> INSTALL -> aspirate from reservoir -> dispense into well
    -> aspirate from well -> waste -> EJECT.
    Because the tip touches the well (during aspiration), a tip change is required for each group.
    """
    out: List[Dict[str, str]] = []

    for start_coord in coord_groups:
        add(out, allowed, 73, "A#ADP_SELECT_4_CHANNEL", "")
        ts, tc, ti = tipper.next_4ch()
        add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(slots["reservoir_slot"], reservoir_col, 1, volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["plate_slot"], start_coord[0], start_coord[1], volume_ul, disp_speed))
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(slots["plate_slot"], start_coord[0], start_coord[1], volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["waste_slot"], 1, 4, volume_ul, 120))
        add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))

    return out


def build_wash_block_1ch_single(allowed, slots, tipper, coord_list, volume_ul, asp_speed, disp_speed, reservoir_col: int):
    """1-channel wash, one round: replace tip for each well.

    Structure: [each well]: SELECT_1CH -> INSTALL -> aspirate from reservoir -> dispense into well
    -> aspirate from well -> waste -> EJECT.
    Because the tip touches the well (during aspiration), a tip change is required for each well.
    """
    out: List[Dict[str, str]] = []

    for coord in coord_list:
        add(out, allowed, 73, "A#ADP_SELECT_1_CHANNEL", "")
        ts, tc, ti = tipper.next_1ch()
        add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(slots["reservoir_slot"], reservoir_col, 1, volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["plate_slot"], coord[0], coord[1], volume_ul, disp_speed))
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(slots["plate_slot"], coord[0], coord[1], volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["waste_slot"], 1, 4, volume_ul, 120))
        add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))

    return out


def build_wash_block_4ch(allowed, slots, tipper, coord_groups, repeats, volume_ul, asp_speed, disp_speed, reservoir_col: int):
    """4-channel-only wash (when there are no 1ch wells): repeat for N rounds."""
    out: List[Dict[str, str]] = []
    for _ in range(max(1, repeats)):
        out += build_wash_block_4ch_single(allowed, slots, tipper, coord_groups, volume_ul, asp_speed, disp_speed, reservoir_col)
    return out


def build_wash_block_1ch(allowed, slots, tipper, coord_list, repeats, volume_ul, asp_speed, disp_speed, reservoir_col: int):
    """1-channel-only wash (when there are no 4ch groups): repeat for N rounds."""
    out: List[Dict[str, str]] = []
    for _ in range(max(1, repeats)):
        out += build_wash_block_1ch_single(allowed, slots, tipper, coord_list, volume_ul, asp_speed, disp_speed, reservoir_col)
    return out


def build_wash_block_mixed(allowed, slots, tipper, coord_groups_4ch, coord_list_1ch, repeats, volume_ul, asp_speed, disp_speed, reservoir_col: int):
    """Combined 4ch+1ch wash - alternating per round: 4ch 1st -> 1ch 1st -> 4ch 2nd -> 1ch 2nd ...

    Structure: [repeat for N rounds]:
      4ch round (replace tip for each group) -> 1ch round (replace tip for each well)
    All wells are washed with uniform timing.
    """
    out: List[Dict[str, str]] = []
    for _ in range(max(1, repeats)):
        if coord_groups_4ch:
            out += build_wash_block_4ch_single(allowed, slots, tipper, coord_groups_4ch, volume_ul, asp_speed, disp_speed, reservoir_col)
        if coord_list_1ch:
            out += build_wash_block_1ch_single(allowed, slots, tipper, coord_list_1ch, volume_ul, asp_speed, disp_speed, reservoir_col)
    return out


def build_remove_block(allowed, slots, tipper, volume_ul, asp_speed, disp_speed):
    """Dedicated (liquid removal) block: INSTALL once -> aspirate from plate -> dispense to waste -> EJECT once."""
    out: List[Dict[str, str]] = []
    ts, tc, ti = tipper.next_install_param()
    add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
    add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",  p(slots["plate_slot"], 1, 1, volume_ul, asp_speed))
    add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE", p(slots["waste_slot"], 1, 4, volume_ul, 120))
    add(out, allowed, 73, "A#ADP_EJECT_PIPETTE",        p(slots["tip_waste_slot"], 0, 0))
    return out
 

def build_remove_block_4ch(allowed, slots, tipper, start_coord, volume_ul, asp_speed, disp_speed):
    """4-channel liquid removal (single group - for backward compatibility)."""
    return build_remove_block_4ch_multi(allowed, slots, tipper, [start_coord], volume_ul, asp_speed, disp_speed)


def build_remove_block_4ch_multi(allowed, slots, tipper, coord_groups, volume_ul, asp_speed, disp_speed):
    """4-channel liquid removal - replace tip for each group.

    Structure: [each group]: SELECT_4CH -> INSTALL -> aspirate from well -> waste -> EJECT.
    Because the tip touches the well (during aspiration), a tip change is required for each group.
    """
    out: List[Dict[str, str]] = []

    for start_coord in coord_groups:
        add(out, allowed, 73, "A#ADP_SELECT_4_CHANNEL", "")
        ts, tc, ti = tipper.next_4ch()
        add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(slots["plate_slot"], start_coord[0], start_coord[1], volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["waste_slot"], 1, 4, volume_ul, 120))
        add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))

    return out


def build_remove_block_1ch(allowed, slots, tipper, coord_list, volume_ul, asp_speed, disp_speed):
    """1-channel liquid removal - replace tip for each well.

    Structure: [each well]: SELECT_1CH -> INSTALL -> aspirate from well -> waste -> EJECT.
    Because the tip touches the well (during aspiration), a tip change is required for each well.
    """
    out: List[Dict[str, str]] = []

    for coord in coord_list:
        add(out, allowed, 73, "A#ADP_SELECT_1_CHANNEL", "")
        ts, tc, ti = tipper.next_1ch()
        add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(slots["plate_slot"], coord[0], coord[1], volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["waste_slot"], 1, 4, volume_ul, 120))
        add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))

    return out

def build_simple_dispense(allowed, slots, tipper, src_slot, src_col, dst_slot, volume_ul, asp_speed, disp_speed):
    """Simple dispense: Reservoir (src_slot=5, src_col) -> plate_slot."""
    out: List[Dict[str, str]] = []
    ts, tc, ti = tipper.next_install_param()
    add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
    add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
        p(src_slot, src_col, 1, volume_ul, asp_speed))
    add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
        p(dst_slot, 1, 1, volume_ul, disp_speed))
    add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))
    return out

def build_dispense_4ch(allowed, slots, tipper, src_slot, src_col, start_coord, volume_ul, asp_speed, disp_speed):
    """4-channel dispense: Reservoir -> Plate (4 wells simultaneously)."""
    out: List[Dict[str, str]] = []
    ts, tc, ti = tipper.next_4ch()
    add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
    add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
        p(src_slot, src_col, 1, volume_ul, asp_speed))
    add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
        p(slots["plate_slot"], start_coord[0], start_coord[1], volume_ul, disp_speed))
    add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))
    return out


def build_dispense_1ch(allowed, slots, tipper, src_slot, src_col, coord_list, volume_ul, asp_speed, disp_speed):
    """1-channel dispense: Reservoir -> Plate (all wells with a single tip)."""
    out: List[Dict[str, str]] = []
    add(out, allowed, 73, "A#ADP_SELECT_1_CHANNEL", "")
    ts, tc, ti = tipper.next_1ch()
    add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
    
    for coord in coord_list:
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(src_slot, src_col, 1, volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["plate_slot"], coord[0], coord[1], volume_ul, disp_speed))
    
    add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))
    return out

def build_row_dispense(allowed, slots, tipper, reservoir_col, row_num, volume_ul, asp_speed, disp_speed, num_cols=8):
    """Per-row variable-volume dispense: dispense the specified volume into all wells of a given row (single-row fallback)."""
    out: List[Dict[str, str]] = []
    src_slot = slots["reservoir_slot"]

    # 4ch groups: grouped in sets of 4 columns (1-4, 5-8, ...)
    groups_4ch = [(row_num, col) for col in range(1, num_cols + 1, 4) if col + 3 <= num_cols]
    # Remaining 1ch wells
    covered = len(groups_4ch) * 4
    wells_1ch = [(row_num, col) for col in range(covered + 1, num_cols + 1)]

    if groups_4ch:
        out += build_dispense_4ch_multi(allowed, slots, tipper, src_slot, reservoir_col,
                                         groups_4ch, volume_ul, asp_speed, disp_speed)
    if wells_1ch:
        out += build_dispense_1ch_multi(allowed, slots, tipper, src_slot, reservoir_col,
                                         wells_1ch, volume_ul, asp_speed, disp_speed)
    return out


def build_row_dispense_batch(allowed, slots, tipper, reservoir_col, row_volumes, asp_speed, disp_speed, num_cols=8):
    """Dispense different volumes into multiple rows from the same reservoir (reusing a single tip).

    row_volumes: [(row_num, volume_ul), ...] - e.g. [(1,15),(2,30),...,(8,120)]
    During dispensing the tip does not touch the well, so reusing one tip for the same reservoir is valid.
    """
    out: List[Dict[str, str]] = []
    plate_slot = slots["plate_slot"]
    src_slot = slots["reservoir_slot"]

    # 4ch group starting columns: [1, 5] (based on 8 columns)
    col_starts_4ch = [c for c in range(1, num_cols + 1, 4) if c + 3 <= num_cols]
    covered = len(col_starts_4ch) * 4
    cols_1ch = list(range(covered + 1, num_cols + 1))

    # 4ch: handle the entire row with a single tip
    if col_starts_4ch:
        add(out, allowed, 73, "A#ADP_SELECT_4_CHANNEL", "")
        ts, tc, ti = tipper.next_4ch()
        add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
        for row_num, vol in row_volumes:
            for col_start in col_starts_4ch:
                add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
                    p(src_slot, reservoir_col, 1, vol, asp_speed))
                add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
                    p(plate_slot, row_num, col_start, vol, disp_speed))
        add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))

    # 1ch remainder: handle the entire row with a single tip
    if cols_1ch:
        add(out, allowed, 73, "A#ADP_SELECT_1_CHANNEL", "")
        ts, tc, ti = tipper.next_1ch()
        add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
        for row_num, vol in row_volumes:
            for col in cols_1ch:
                add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
                    p(src_slot, reservoir_col, 1, vol, asp_speed))
                add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
                    p(plate_slot, row_num, col, vol, disp_speed))
        add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))

    return out


def build_dispense_4ch_multi(allowed, slots, tipper, src_slot, src_col, coord_groups, volume_ul, asp_speed, disp_speed):
    """4-channel dispense: handle all groups with a single tip."""
    out: List[Dict[str, str]] = []
    add(out, allowed, 73, "A#ADP_SELECT_4_CHANNEL", "")
    ts, tc, ti = tipper.next_4ch()
    add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
    
    for start_coord in coord_groups:
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(src_slot, src_col, 1, volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["plate_slot"], start_coord[0], start_coord[1], volume_ul, disp_speed))
    
    add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))
    return out

def build_dispense_1ch_multi(allowed, slots, tipper, src_slot, src_col, coord_list, volume_ul, asp_speed, disp_speed):
    """1-channel dispense: handle all wells with a single tip."""
    out: List[Dict[str, str]] = []
    add(out, allowed, 73, "A#ADP_SELECT_1_CHANNEL", "")
    ts, tc, ti = tipper.next_1ch()
    add(out, allowed, 73, "A#ADP_INSTALL_PIPETTE", p(ts, tc, ti))
    
    for coord in coord_list:
        add(out, allowed, 73, "A#ADP_ASPIRATE_FROM_PLATE",
            p(src_slot, src_col, 1, volume_ul, asp_speed))
        add(out, allowed, 73, "A#ADP_DISPENSE_INTO_PLATE",
            p(slots["plate_slot"], coord[0], coord[1], volume_ul, disp_speed))
    
    add(out, allowed, 73, "A#ADP_EJECT_PIPETTE", p(slots["tip_waste_slot"], 0, 0))
    return out

def build_hotplate_incubation(allowed, slots, temp_c: int, minutes: int):
    """Incubation/reaction: TRANS plate to hotplate_slot -> TEMPPLATE_ON(temp) -> WAIT_MOTION(sec) -> return to plate_slot."""
    out: List[Dict[str, str]] = []
    if minutes <= 0:
        add(out, allowed, 73, "A#TRANS", f"({slots['plate_slot']},{slots['hotplate_slot']})")
        add(out, allowed, 73, "A#TEMPPLATE_ON", f"({temp_c})")
        add(out, allowed, 73, "A#TRANS", f"({slots['hotplate_slot']},{slots['plate_slot']})")
        return out
    sec = minutes * 60
    add(out, allowed, 73, "A#TRANS",         p(slots["plate_slot"],     slots["hotplate_slot"]))
    add(out, allowed, 73, "A#TEMPPLATE_ON",  p(temp_c))
    add(out, allowed, 73, "A#WAIT_MOTION",   p(sec))
    add(out, allowed, 73, "A#TRANS",         p(slots["hotplate_slot"],  slots["plate_slot"]))
    return out


def build_static_wait(allowed, slots, minutes: int):
    """Room-temperature wait: move plate to hotplate (deck 3) -> set room temperature -> wait -> return."""
    out: List[Dict[str, str]] = []
    sec = max(minutes, 1) * 60
    add(out, allowed, 73, "A#TRANS",        p(slots["plate_slot"], slots["hotplate_slot"]))
    add(out, allowed, 73, "A#TEMPPLATE_ON", p(20))  # room temperature 20 C
    add(out, allowed, 73, "A#WAIT_MOTION",  p(sec))
    add(out, allowed, 73, "A#TRANS",        p(slots["hotplate_slot"], slots["plate_slot"]))
    return out


def build_shake(allowed, slots, speed: int, time_sec: int):
    """Shaking block."""
    out: List[Dict[str, str]] = []
    add(out, allowed, 73, "A#TRANS",       p(slots["plate_slot"], slots["shaker_slot"]))
    add(out, allowed, 73, "A#SHAKE_START", p(speed, time_sec))
    add(out, allowed, 73, "A#TRANS",       p(slots["shaker_slot"], slots["plate_slot"]))
    return out


def build_cap_open_close(allowed, slots, open_: bool):
    """CAP open/close block."""
    out: List[Dict[str, str]] = []
    if open_:
        add(out, allowed, 73, "A#CAP_OPEN",  f"({slots['plate_slot']},{slots['cap_parking_slot']})")
    else:
        add(out, allowed, 73, "A#CAP_CLOSE", f"({slots['plate_slot']},{slots['cap_parking_slot']})")
    return out


def build_elisa_readout_block(allowed, slots):
    """Dedicated ELISA readout (OD) block."""
    out: List[Dict[str, str]] = []
    add(out, allowed, 73, "A#CAP_CLOSE", "1 9")
    add(out, allowed, 74, "A#GET", "4 1")
    add(out, allowed, 74, "A#PUT", "5 1")
    add(out, allowed, 75, "A#CAP_OPEN", "1 2")
    add(out, allowed, 75, "A#ANALYZER_OPEN", "")
    add(out, allowed, 75, "A#TRANS", "1 3")
    add(out, allowed, 75, "A#ANALYZER_CLOSE", "")
    add(out, allowed, 75, "A#ANALYZER_START", "json file name")
    add(out, allowed, 75, "A#ANALYZER_OPEN", "")
    add(out, allowed, 75, "A#TRANS", "3 1")
    add(out, allowed, 75, "A#ANALYZER_CLOSE", "")
    add(out, allowed, 75, "A#CAP_CLOSE", "1 2")
    add(out, allowed, 74, "A#GET", "5 1")
    add(out, allowed, 74, "A#PUT", "2 1")
    return out

def line_to_sequence(line: str, allowed, slots, tipper, defaults: Dict[str, int], wellplate_coords: Dict = None) -> List[Dict[str, str]]:
    """Natural language -> sequence (with well coordinate recognition)."""
    out: List[Dict[str, str]] = []
    D = defaults

    # User-performed step - not included in the automation sequence
    if re.search(r"사용자\s*수행\s*선행|사용자가\s*직접|manual\s*step|user[\s\-]*performed", line, flags=re.I):
        print(f"[SKIP] User-performed step: {line}")
        return []

    print(f"[DEBUG] wellplate_coords provided: {wellplate_coords is not None}")
    if wellplate_coords:
        print(f"[DEBUG] wellplate_coords content: {wellplate_coords}")

    channel = None
    start_coord = None
    coord_list = []

    if wellplate_coords:
        coords = wellplate_coords.get('ALL', {})

        print(f"[DEBUG] Available coordinates: {coords}")

        all_4ch_groups = coords.get('4ch_groups', [])
        all_1ch_wells = coords.get('1ch_wells', [])

        print(f"[DEBUG] 4-channel groups: {all_4ch_groups}")
        print(f"[DEBUG] 1-channel wells: {all_1ch_wells}")

        if all_4ch_groups or all_1ch_wells:
            channel = 'multi'
            start_coord = None
            coord_list = []
            print(f"[DEBUG] Multi-processing mode enabled")

    else:
        channel_match = re.search(r'\[(\d)ch', line, flags=re.I)
        coord_match = re.search(r'시작좌표\s*\((\d+),(\d+)\)', line, flags=re.I)
        coord_list_match = re.search(r'좌표\s*\((.+?)\)', line, flags=re.I)
        
        channel = int(channel_match.group(1)) if channel_match else None
        start_coord = (int(coord_match.group(1)), int(coord_match.group(2))) if coord_match else None
        
        if coord_list_match:
            coord_str = coord_list_match.group(1)
            for coord_pair in coord_str.split(','):
                if '(' in coord_pair:
                    nums = re.findall(r'\d+', coord_pair)
                    if len(nums) >= 2:
                        coord_list.append((int(nums[0]), int(nums[1])))

    # Extract "세척 4회", "4회", "wash 4x" - prefer "회" to avoid mismatching "1X" (reagent concentration)
    _rep_m = re.search(r"(\d+)\s*회", line) or re.search(r"세척\s*(\d+)\s*[xX×]", line, flags=re.I)
    repeats = int(_rep_m.group(1)) if _rep_m else 1
    vol_ul = extract_volume_ul(line, default_ul=D["volume_ul"])
    asp_speed = D["aspirate_speed"]
    disp_speed = D["dispense_speed"]
    inc_min = extract_minutes(line, default_min=D["incubation_min"])
    temp_c = extract_int(r"(\d+)\s*°?c", line, default=D["hotplate_temp"])

    # CAP open/close
    if re.search(r"뚜껑\s*열|cap\s*open", line, flags=re.I):
        return build_cap_open_close(allowed, slots, True)
    if re.search(r"뚜껑\s*닫|cap\s*close", line, flags=re.I):
        return build_cap_open_close(allowed, slots, False)

    # Tap / gentle mix / short shaking -> replaced with short shaking (200 rpm, default 10 sec)
    # If a time is specified, use that time (e.g. "tap for 1 minute" -> 60 sec, "30초" -> 30 sec)
    if re.search(r"(gently\s*tap|tap\s*the\s*plate|가볍게\s*두드|톡톡|gentle\s*mix|부드럽게\s*혼합|가볍게\s*섞|짧은\s*shaking)", line, flags=re.I):
        sec_m = re.search(r"(\d+)\s*초", line)
        if sec_m:
            tap_sec = int(sec_m.group(1))
        elif inc_min > 0:
            tap_sec = inc_min * 60
        else:
            tap_sec = 10
        return build_shake(allowed, slots, 200, tap_sec)

    # "Move plate to shaker" - ignore lines that only instruct a move (build_shake already includes TRANS)
    if re.search(r"(?:shaker|셰이커).*(?:이동|옮|transfer)", line, flags=re.I) and not re.search(r"(rpm|shake_start|분주|시간|분\b)", line, flags=re.I):
        return []  # skip - the full TRANS+SHAKE is handled at the next SHAKE_START line

    # Standalone WAIT_MOTION line (when the model emits "WAIT_MOTION 1800" or "WAIT_MOTION 30분" as a separate line)
    if re.search(r"WAIT_MOTION", line, flags=re.I):
        # First, check whether a "분" or "min" unit is present
        wait_min = extract_minutes(line, default_min=-1)
        if wait_min > 0:
            sec = wait_min * 60
        else:
            # If only a number is present, as in "WAIT_MOTION 1800" -> use it directly as seconds
            sec_m = re.search(r"WAIT_MOTION\s+(\d+)", line, flags=re.I)
            if sec_m:
                sec = int(sec_m.group(1))
            else:
                sec = 60  # fallback
        out: List[Dict[str, str]] = []
        add(out, allowed, 73, "A#WAIT_MOTION", p(sec))
        return out

    # Standalone TRANS line - ignore (build_static_wait/build_hotplate_incubation already include TRANS,
    # so a TRANS emitted as a separate line is skipped to avoid duplicate generation)
    if re.search(r"^\s*-?\s*TRANS\s+\d+\s*[→\->]+\s*\d+\s*$", line, flags=re.I):
        return []  # skip

    # Standalone TEMPPLATE_ON line - ignore (build_static_wait/build_hotplate_incubation already include TEMPPLATE_ON)
    if re.search(r"^\s*-?\s*TEMPPLATE_ON\s+\d+\s*$", line, flags=re.I):
        return []  # skip

    # Standalone SHAKE_START line (when the model emits "SHAKE_START 500 rpm" as a separate line)
    if re.search(r"SHAKE_START", line, flags=re.I):
        rpm = extract_rpm(line, D["shake_speed"])
        _sec_m = re.search(r"(\d+)\s*초", line)
        secs = int(_sec_m.group(1)) if _sec_m else (inc_min * 60 if inc_min > 0 else D["shake_time_sec"])
        return build_shake(allowed, slots, rpm, secs)

    # Shaker + Incubation simultaneously
    if re.search(r"(incubat|배양)", line, flags=re.I) and re.search(r"(shake|shaker|흔들|교반|rpm)", line, flags=re.I):
        rpm = extract_rpm(line, D["shake_speed"])
        _sec_m = re.search(r"(\d+)\s*초", line)
        secs = int(_sec_m.group(1)) if _sec_m else (inc_min * 60 if inc_min > 0 else D["shake_time_sec"])
        return build_shake(allowed, slots, rpm, secs)

    # Shaker only
    if re.search(r"(shake|shaker|흔들|교반|rpm)", line, flags=re.I) and not re.search(r"(incubat|배양)", line, flags=re.I):
        rpm = extract_rpm(line, D["shake_speed"])
        _sec_m = re.search(r"(\d+)\s*초", line)
        secs = int(_sec_m.group(1)) if _sec_m else (inc_min * 60 if inc_min > 0 else D["shake_time_sec"])
        return build_shake(allowed, slots, rpm, secs)

    # Static wait: room-temperature/benchtop incubation or WAIT + room temperature (wait in place without moving the plate)
    if re.search(r"상온|room\s*temp|benchtop|벤치탑|암조건", line, flags=re.I) and re.search(r"배양|incubat|WAIT|대기|방치", line, flags=re.I) and not re.search(r"shake|shaker|흔들|교반|rpm", line, flags=re.I):
        return build_static_wait(allowed, slots, inc_min)

    # Hotplate/Incubation (temperature-specified incubation - when not at room temperature)
    if re.search(r"incubat|배양|반응|hotplate|온도", line, flags=re.I) and not re.search(r"분주|dispense|aspirate|흡인|상온|벤치탑|benchtop|room\s*temp", line, flags=re.I):
        return build_hotplate_incubation(allowed, slots, temp_c, inc_min)

    # Liquid removal (includes aspirate/decant - the "Aspirate the wells to remove liquid" pattern from PDFs)
    if re.search(r"용액\s*제거|remove\s*(liquid|solution)|제거\s*합니다|이전.*단계.*용액.*제거|aspirate\s*(the\s*)?wells|decant", line, flags=re.I):
        rem_vol = vol_ul if vol_ul else 100
        if channel == 'multi' and wellplate_coords:
            coords = wellplate_coords.get('ALL', {})

            # 4-channel groups: handle all groups with a single tip
            if coords.get('4ch_groups'):
                out += build_remove_block_4ch_multi(allowed, slots, tipper, coords['4ch_groups'], rem_vol, asp_speed, disp_speed)

            # 1-channel wells: handle all wells with a single tip
            if coords.get('1ch_wells'):
                out += build_remove_block_1ch(allowed, slots, tipper, coords['1ch_wells'], rem_vol, asp_speed, disp_speed)
            return out

    # Wash
    if re.search(r"세척", line, flags=re.I) and re.search(r"\d+\s*[회xX×]", line):
        rcol = parse_reservoir_col(line, 1)
        if channel == 'multi' and wellplate_coords:
            coords = wellplate_coords.get('ALL', {})
            groups_4ch = coords.get('4ch_groups', [])
            wells_1ch = coords.get('1ch_wells', [])

            # If both 4ch and 1ch are present, alternate per round (4ch 1st -> 1ch 1st -> 4ch 2nd -> 1ch 2nd ...)
            if groups_4ch and wells_1ch:
                out += build_wash_block_mixed(allowed, slots, tipper, groups_4ch, wells_1ch, repeats, vol_ul or 300, asp_speed, disp_speed, rcol)
            elif groups_4ch:
                out += build_wash_block_4ch(allowed, slots, tipper, groups_4ch, repeats, vol_ul or 300, asp_speed, disp_speed, rcol)
            elif wells_1ch:
                out += build_wash_block_1ch(allowed, slots, tipper, wells_1ch, repeats, vol_ul or 300, asp_speed, disp_speed, rcol)
            return out
        elif channel == 4 and start_coord:
            return build_wash_block_4ch(allowed, slots, tipper, start_coord, repeats, vol_ul or 300, asp_speed, disp_speed, rcol)
        elif channel == 1 and coord_list:
            return build_wash_block_1ch(allowed, slots, tipper, coord_list, repeats, vol_ul or 300, asp_speed, disp_speed, rcol)
        else:
            return build_wash_block(allowed, slots, tipper, repeats, vol_ul or 300, asp_speed, disp_speed, rcol)

    # ── Type-specific dispense trigger ──
    # When type-distinguishing keywords such as "standard or sample" or "표준액 또는 검체" appear together,
    # dispense into the BLANK/CALIBRANT/SAMPLE coordinates from their respective reservoirs
    _type_trigger = re.search(
        r"(standard\s*(or|,)\s*sample|sample\s*(or|,)\s*standard"
        r"|표준[액물질]?\s*(또는|[,/]|및)\s*(검체|샘플|시료)"
        r"|(검체|샘플|시료)\s*(또는|[,/]|및)\s*표준[액물질]?"
        r"|standard\s*,\s*control\s*,?\s*(or|and)?\s*sample"
        r"|시료[/]?표준액\s*분주|첫\s*분주.*타입\s*별)",
        line, flags=re.I
    )
    if _type_trigger and channel == 'multi' and wellplate_coords:
        print(f"[TYPE-DISPENSE] Type-specific dispense detected: {line}")
        vol = vol_ul or 100

        # ── Per-solution-group dispense (used when solution_group_coords is available) ──
        # Use solution_group_coords/solution_reservoir_map passed from the SequenceBuilder instance
        _sg_coords = wellplate_coords.get('_solution_group_coords', {})
        _sg_reservoir = wellplate_coords.get('_solution_reservoir_map', {})

        if _sg_coords and _sg_reservoir:
            print(f"[TYPE-DISPENSE] Per-solution-group dispense mode: {len(_sg_coords)} groups")

            # 4ch first: iterate over all solution groups (Reservoir Col order = the order loaded by the operator)
            sorted_groups = sorted(_sg_reservoir.items(), key=lambda x: x[1])

            has_any_4ch = any(
                _sg_coords.get(gk, {}).get('4ch_groups', [])
                for gk, _ in sorted_groups
            )
            if has_any_4ch:
                for group_key, rcol in sorted_groups:
                    sg = _sg_coords.get(group_key, {})
                    t_4ch = sg.get('4ch_groups', [])
                    if not t_4ch:
                        continue
                    print(f"[TYPE-DISPENSE] {group_key} 4ch: Reservoir Col {rcol}, groups={len(t_4ch)}")
                    # Same solution = reuse the same tip (build_dispense_4ch_multi handles all groups with one tip)
                    out += build_dispense_4ch_multi(
                        allowed, slots, tipper, slots["reservoir_slot"], rcol,
                        t_4ch, vol, asp_speed, disp_speed)

            # 1ch next
            has_any_1ch = any(
                _sg_coords.get(gk, {}).get('1ch_wells', [])
                for gk, _ in sorted_groups
            )
            if has_any_1ch:
                for group_key, rcol in sorted_groups:
                    sg = _sg_coords.get(group_key, {})
                    t_1ch = sg.get('1ch_wells', [])
                    if not t_1ch:
                        continue
                    print(f"[TYPE-DISPENSE] {group_key} 1ch: Reservoir Col {rcol}, wells={len(t_1ch)}")
                    out += build_dispense_1ch_multi(
                        allowed, slots, tipper, slots["reservoir_slot"], rcol,
                        t_1ch, vol, asp_speed, disp_speed)

        else:
            # Fallback: legacy per-type dispense (when there is no solution_group)
            type_reservoir_map = {'BLANK': 1, 'CALIBRANT': 2, 'SAMPLE': 3}

            for tkey, default_rcol in type_reservoir_map.items():
                type_pattern = {
                    'BLANK': r"(?:blank|영점|zero\s*standard|diluent).*?reservoir\s*(\d+)",
                    'CALIBRANT': r"(?:calibrant|standard|표준|calibrator).*?reservoir\s*(\d+)",
                    'SAMPLE': r"(?:sample|검체|시료|샘플).*?reservoir\s*(\d+)",
                }
                m_rcol = re.search(type_pattern[tkey], line, flags=re.I)
                if m_rcol:
                    type_reservoir_map[tkey] = int(m_rcol.group(1))

            has_any_4ch = False
            for type_key in ['BLANK', 'CALIBRANT', 'SAMPLE']:
                type_coords = wellplate_coords.get(type_key, {})
                t_4ch = type_coords.get('4ch_groups', [])
                if t_4ch:
                    has_any_4ch = True
                    break

            if has_any_4ch:
                for type_key in ['BLANK', 'CALIBRANT', 'SAMPLE']:
                    type_coords = wellplate_coords.get(type_key, {})
                    t_4ch = type_coords.get('4ch_groups', [])
                    if not t_4ch:
                        continue
                    rcol = type_reservoir_map[type_key]
                    print(f"[TYPE-DISPENSE] {type_key} 4ch: Reservoir {rcol}, groups={len(t_4ch)}")
                    out += build_dispense_4ch_multi(
                        allowed, slots, tipper, slots["reservoir_slot"], rcol,
                        t_4ch, vol, asp_speed, disp_speed)

            has_any_1ch = False
            for type_key in ['BLANK', 'CALIBRANT', 'SAMPLE']:
                type_coords = wellplate_coords.get(type_key, {})
                t_1ch = type_coords.get('1ch_wells', [])
                if t_1ch:
                    has_any_1ch = True
                    break

            if has_any_1ch:
                for type_key in ['BLANK', 'CALIBRANT', 'SAMPLE']:
                    type_coords = wellplate_coords.get(type_key, {})
                    t_1ch = type_coords.get('1ch_wells', [])
                    if not t_1ch:
                        continue
                    rcol = type_reservoir_map[type_key]
                    print(f"[TYPE-DISPENSE] {type_key} 1ch: Reservoir {rcol}, wells={len(t_1ch)}")
                    out += build_dispense_1ch_multi(
                        allowed, slots, tipper, slots["reservoir_slot"], rcol,
                        t_1ch, vol, asp_speed, disp_speed)

        # Check for incubation after dispensing
        if re.search(r"배양|incubat", line, flags=re.I) and inc_min > 0:
            temp = 20 if re.search(r"상온", line) else extract_int(r"(\d+)\s*°?c", line, default=37)
            out += build_hotplate_incubation(allowed, slots, temp, inc_min)

        return out

    # ── Per-row variable-volume dispense (e.g. "Reservoir 4 (BSA) Row 1에 15 µL 분주") ──
    _row_m = re.search(r"reservoir\s*(\d+).*?row\s*(\d+).*?(\d+)\s*[µu]?l.*?분주", line, flags=re.I)
    if _row_m:
        rcol = int(_row_m.group(1))
        row_num = int(_row_m.group(2))
        vol = int(_row_m.group(3))
        print(f"[ROW-DISPENSE] Reservoir {rcol}, Row {row_num}, {vol} uL")
        return build_row_dispense(allowed, slots, tipper, rcol, row_num, vol, asp_speed, disp_speed)

    # Generic reservoir dispense pattern (number-based)
    if re.search(r"reservoir\s*\d+.*?\d+\s*µ?L.*?분주", line, flags=re.I):
        rcol = parse_reservoir_col(line)
        vol = vol_ul or 100

        if channel == 'multi' and wellplate_coords:
            # Based on UI drag information: if wells are mapped to rcol, dispense only into those wells; otherwise into all wells
            _rcol_map = wellplate_coords.get('_rcol_to_coords', {})
            if rcol in _rcol_map:
                coords = _rcol_map[rcol]
                print(f"[TYPE-DISPENSE] Reservoir {rcol} -> dispense into specified wells only")
            else:
                coords = wellplate_coords.get('ALL', {})

            if coords.get('4ch_groups'):
                out += build_dispense_4ch_multi(allowed, slots, tipper, slots["reservoir_slot"], rcol, coords['4ch_groups'], vol, asp_speed, disp_speed)

            if coords.get('1ch_wells'):
                out += build_dispense_1ch_multi(allowed, slots, tipper, slots["reservoir_slot"], rcol, coords['1ch_wells'], vol, asp_speed, disp_speed)
        else:
            out += build_simple_dispense(allowed, slots, tipper, slots["reservoir_slot"], rcol, slots["plate_slot"], vol, asp_speed, disp_speed)

        # Check for incubation after dispensing
        if re.search(r"배양|incubat", line, flags=re.I):
            inc_min = extract_minutes(line, default_min=0)
            if inc_min > 0:
                temp = 20 if re.search(r"상온", line) else extract_int(r"(\d+)\s*°?c", line, default=37)
                out += build_hotplate_incubation(allowed, slots, temp, inc_min)

        return out

    # Reservoir dispense
    if re.search(r"(reservoir\s*\d+.*분주|\d+\s*µ?µ?l.*분주|dispense.*reservoir|reservoir.*dispense)", line, flags=re.I):
        rcol = parse_reservoir_col(line)
        vol = vol_ul or 100

        print(f"[DEBUG] Dispense detected: vol={vol}, rcol={rcol}, channel={channel}")

        if channel == 'multi' and wellplate_coords:
            # Based on UI drag information: if wells are mapped to rcol, dispense only into those wells; otherwise into all wells
            _rcol_map = wellplate_coords.get('_rcol_to_coords', {})
            if rcol in _rcol_map:
                coords = _rcol_map[rcol]
                print(f"[TYPE-DISPENSE] Reservoir {rcol} -> dispense into specified wells only")
            else:
                coords = wellplate_coords.get('ALL', {})

            if coords.get('4ch_groups'):
                out += build_dispense_4ch_multi(allowed, slots, tipper, slots["reservoir_slot"], rcol, coords['4ch_groups'], vol, asp_speed, disp_speed)

            if coords.get('1ch_wells'):
                out += build_dispense_1ch_multi(allowed, slots, tipper, slots["reservoir_slot"], rcol, coords['1ch_wells'], vol, asp_speed, disp_speed)

        elif channel == 4 and start_coord:
            out += build_dispense_4ch(allowed, slots, tipper, slots["reservoir_slot"], rcol, start_coord, vol, asp_speed, disp_speed)
        elif channel == 1 and coord_list:
            out += build_dispense_1ch(allowed, slots, tipper, slots["reservoir_slot"], rcol, coord_list, vol, asp_speed, disp_speed)
        else:
            out += build_simple_dispense(allowed, slots, tipper, slots["reservoir_slot"], rcol, slots["plate_slot"], vol, asp_speed, disp_speed)

        if re.search(r"incubat|배양|반응", line, flags=re.I) and inc_min > 0:
            out += build_hotplate_incubation(allowed, slots, temp_c, inc_min)
        return out

    # OD measurement
    if re.search(r"(?:\bO\.?\s*D\.?\b|OD측정|흡광|판독|plate\s*reader|플레이트\s*리더|450\s*nm|620\s*nm)", line, flags=re.I):
        return []

    # QC / quantification range
    if re.search(r"(정량\s*범위|QC|품질|standard\s*curve|표준\s*곡선)", line, flags=re.I):
        return []

    return [{"Handler": "#", "Command": "NOT_MAPPED", "Input Parameters": line}]


class SequenceBuilder:
    """Sequence builder."""

    def __init__(self, config, wellplate_text=None):
        self.config = config
        self.wellplate_text = wellplate_text
        self.wellplate_coords = self._parse_wellplate_coords() if wellplate_text else {}
        # Inject solution-group coordinates into wellplate_coords (so line_to_sequence can access them)
        if hasattr(self, 'solution_group_coords') and self.solution_group_coords:
            self.wellplate_coords['_solution_group_coords'] = self.solution_group_coords
            self.wellplate_coords['_solution_reservoir_map'] = self.solution_reservoir_map
            # Direct reservoir_col -> coordinate mapping (based on UI drag information)
            rcol_to_coords = {}
            for gk, rcol in self.solution_reservoir_map.items():
                rcol_to_coords[rcol] = self.solution_group_coords.get(gk, {'4ch_groups': [], '1ch_wells': []})
            self.wellplate_coords['_rcol_to_coords'] = rcol_to_coords
                            
    def _parse_wellplate_coords(self):
        """Extract coordinates from the Well Plate Layout text (ALL + per-type + per-solution-group)."""
        coords = {
            'ALL': {'4ch_groups': [], '1ch_wells': []},
            'BLANK': {'4ch_groups': [], '1ch_wells': []},
            'CALIBRANT': {'4ch_groups': [], '1ch_wells': []},
            'SAMPLE': {'4ch_groups': [], '1ch_wells': []},
        }

        # Per-solution-group coordinates + reservoir mapping
        self.solution_group_coords = {}  # key: 'BLANK_0', 'CALIBRANT_0', 'SAMPLE_0', ...
        self.solution_reservoir_map = {}  # key -> reservoir_col

        if not self.wellplate_text:
            return coords

        lines = self.wellplate_text.split('\n')

        current_section = None  # 'ALL_4CH', 'ALL_1CH', 'BLANK', 'CALIBRANT', 'SAMPLE', 'SOLUTION_COORDS'
        current_type_key = None
        current_solution_key = None

        for line in lines:
            line = line.strip()

            # Existing global sections
            if '[4-Channel Groups]' in line:
                current_section = 'ALL_4CH'
                continue
            elif '[1-Channel Wells]' in line:
                current_section = 'ALL_1CH'
                continue
            # Per-type sections
            elif '[BLANK Wells]' in line:
                current_section = 'TYPE'
                current_type_key = 'BLANK'
                continue
            elif '[CALIBRANT Wells]' in line:
                current_section = 'TYPE'
                current_type_key = 'CALIBRANT'
                continue
            elif '[SAMPLE Wells]' in line:
                current_section = 'TYPE'
                current_type_key = 'SAMPLE'
                continue
            # Solution-group coordinates section
            elif '[Solution Groups - Coordinates]' in line:
                current_section = 'SOLUTION_COORDS'
                continue
            elif '[Solution Groups]' in line:
                current_section = 'SOLUTION_LIST'
                continue
            elif '[Reservoir Loading Guide]' in line:
                current_section = 'RESERVOIR_GUIDE'
                continue
            elif '[Type Configuration]' in line:
                current_section = 'TYPE_CONFIG'
                continue
            elif '[Pipetting Strategy]' in line:
                break

            # Parse global 4ch groups
            if current_section == 'ALL_4CH' and 'Start:' in line:
                match = re.search(r'Start:\s*\((\d+),(\d+)\)', line)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    coords['ALL']['4ch_groups'].append((x, y))
                    print(f"[DEBUG] 4-channel group added: ({x},{y})")

            # Parse global 1ch wells
            if current_section == 'ALL_1CH' and ' at (' in line:
                match = re.search(r'([A-H]\d+)\s+at\s*\((\d+),(\d+)\)', line)
                if match:
                    well_name = match.group(1)
                    x = int(match.group(2))
                    y = int(match.group(3))
                    coords['ALL']['1ch_wells'].append((x, y))
                    print(f"[DEBUG] 1-channel well added: {well_name} at ({x},{y})")

            # Parse per-type sections
            if current_section == 'TYPE' and current_type_key:
                if 'Start:' in line:
                    match = re.search(r'Start:\s*\((\d+),(\d+)\)', line)
                    if match:
                        x = int(match.group(1))
                        y = int(match.group(2))
                        coords[current_type_key]['4ch_groups'].append((x, y))
                        print(f"[DEBUG] {current_type_key} 4-channel group: ({x},{y})")
                elif ' at (' in line:
                    match = re.search(r'([A-H]\d+)\s+at\s*\((\d+),(\d+)\)', line)
                    if match:
                        well_name = match.group(1)
                        x = int(match.group(2))
                        y = int(match.group(3))
                        coords[current_type_key]['1ch_wells'].append((x, y))
                        print(f"[DEBUG] {current_type_key} 1-channel well: {well_name} at ({x},{y})")

            # ── Parse per-solution-group coordinates ──
            if current_section == 'SOLUTION_COORDS':
                # Detect a group key in the "  BLANK_0: Reservoir Col 1" format
                key_match = re.match(r'\s*((?:BLANK|CALIBRANT|SAMPLE)_\d+):\s*Reservoir\s*Col\s*(\d+)', line)
                if key_match:
                    current_solution_key = key_match.group(1)
                    rcol = int(key_match.group(2))
                    self.solution_reservoir_map[current_solution_key] = rcol
                    if current_solution_key not in self.solution_group_coords:
                        self.solution_group_coords[current_solution_key] = {'4ch_groups': [], '1ch_wells': []}
                    print(f"[DEBUG] Solution group: {current_solution_key} -> Reservoir Col {rcol}")
                    continue

                # "    4ch_groups: [(1,1), (2,1)]" format
                if current_solution_key and '4ch_groups:' in line:
                    for m in re.finditer(r'\((\d+),(\d+)\)', line):
                        x, y = int(m.group(1)), int(m.group(2))
                        self.solution_group_coords[current_solution_key]['4ch_groups'].append((x, y))
                        print(f"[DEBUG] {current_solution_key} 4ch: ({x},{y})")
                    continue

                # "    1ch_wells: [(3,5), (3,6)]" format
                if current_solution_key and '1ch_wells:' in line:
                    for m in re.finditer(r'\((\d+),(\d+)\)', line):
                        x, y = int(m.group(1)), int(m.group(2))
                        self.solution_group_coords[current_solution_key]['1ch_wells'].append((x, y))
                        print(f"[DEBUG] {current_solution_key} 1ch: ({x},{y})")
                    continue

            # Also parse the reservoir mapping from the [Solution Groups] section
            if current_section == 'SOLUTION_LIST':
                sg_match = re.match(r'\s*((?:BLANK|CALIBRANT|SAMPLE)_\d+):.*Reservoir\s*Col\s*(\d+)', line)
                if sg_match:
                    sg_key = sg_match.group(1)
                    sg_rcol = int(sg_match.group(2))
                    self.solution_reservoir_map[sg_key] = sg_rcol

        print(f"[DEBUG] Final parsing result: {coords}")
        if self.solution_group_coords:
            print(f"[DEBUG] Solution group coords: {self.solution_group_coords}")
            print(f"[DEBUG] Solution reservoir map: {self.solution_reservoir_map}")
        return coords
                        
    def build_sequences(self):
        """Build sequences from all structured protocols."""
        self.config.ensure_output_dirs()

        protocol_files = list(self.config.OUTPUT_DIR.glob(f"*{self.config.STRUCTURED_SUFFIX}"))
        if not protocol_files:
            print(f"[WARN] No structured protocol files found: {self.config.OUTPUT_DIR}")
            return

        print(f"Converting {len(protocol_files)} protocols into sequences.")

        for protocol_path in sorted(protocol_files):
            self.build_single_sequence(protocol_path)

    def build_single_sequence(self, txt_path: Path):
        """Build a sequence from a single protocol."""
        print(f"[DEBUG] wellplate_coords present: {bool(self.wellplate_coords)}")
        if self.wellplate_coords:
            print(f"[DEBUG] wellplate_coords content: {self.wellplate_coords}")
        else:
            print(f"[DEBUG] wellplate_text present: {bool(self.wellplate_text)}")
            if self.wellplate_text:
                print(f"[DEBUG] wellplate_text excerpt: {self.wellplate_text[:200]}...")

        sheets = load_all_sheets(self.config.EXCEL_PATH)
        allowed_cmds = get_allowed_commands(sheets, self.config.COMMAND_WHITELIST)
        if not allowed_cmds:
            raise RuntimeError("No allowed commands available. Please check the Command sheet.")
        slots = infer_deck_slots(sheets, self.config.DEFAULT_SLOTS)

        txt = txt_path.read_text(encoding="utf-8", errors="ignore")

        m = re.search(r"<INSTRUMENT>(.*?)</INSTRUMENT>", txt, flags=re.S|re.I)
        instr_text = m.group(1) if m else ""

        detected = ImpossibleActionDetector.detect(instr_text)
        if detected:
            msg = (
                "Non-automatable protocol detected.\n"
                "The following steps cannot be performed by the Bioforge instrument:\n"
                + "\n".join(f"- {kw}" for kw in detected)
            )
            print(msg)
            # Only print a warning and continue building the sequence (to avoid interrupting batch processing)

        m5 = re.search(r"\[\s*5\s*\][^\n]*\n(.*?)(?=\n\[\s*6\s*\]|$)", instr_text, flags=re.S|re.I)
        section5 = m5.group(1) if m5 else ""
        lines = [ln.strip() for ln in section5.splitlines() if ln.strip()]

        m6 = re.search(r"\[\s*6\s*\][^\n]*\n(.*)$", instr_text, flags=re.S|re.I)
        section6 = m6.group(1) if m6 else ""

        # Temperature extraction: "37°C", "20 °C", etc. - the ° (degree) symbol is required to avoid mismatching cases like "6 Conjugate"
        m_temp = re.search(r"(\d+)\s*°\s*C", " ".join(lines))
        global_temp_c = int(m_temp.group(1)) if m_temp else self.config.DEFAULTS["hotplate_temp"]

        tipper = TipAllocator(slots["tip_slot"])
        rows: List[Dict[str, str]] = []

        # Initial setup commands
        rows.append(cmd(73, "A#TEMPPLATE_ON", f"{global_temp_c}"))
        rows.append(cmd(74, "A#GET", "1 1"))
        rows.append(cmd(74, "A#PUT", "4 1"))
        rows.append(cmd(73, "A#CAP_OPEN", "1 9"))

        # Channel mode selection is handled within each block function (build_remove_block_4ch_multi,
        # build_wash_block_4ch, etc.), which include SELECT_4_CHANNEL/SELECT_1_CHANNEL themselves, so it is not added here.

        # ── Preprocessing: merge split Shaker lines ──
        # When the model splits "Move to shaker" / "SHAKE_START 500 rpm" / "WAIT_MOTION 120분"
        # into 3 lines -> merge into a single line "Shaker 500 rpm, 120분 배양"
        merged_lines = []
        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            # Detect the "Move to shaker" pattern (move-only instruction, no rpm/time)
            if re.search(r"(?:shaker|셰이커).*(?:이동|옮|transfer)", ln, flags=re.I) and not re.search(r"\d+\s*rpm|\d+\s*분", ln, flags=re.I):
                # Collect SHAKE_START / WAIT_MOTION from the following lines
                rpm_val = None
                wait_min = None
                j = i + 1
                while j < len(lines) and j <= i + 2:
                    next_ln = lines[j].strip()
                    if re.search(r"SHAKE_START", next_ln, flags=re.I):
                        m_rpm = re.search(r"(\d+)\s*rpm", next_ln, flags=re.I)
                        if m_rpm:
                            rpm_val = m_rpm.group(1)
                        j += 1
                    elif re.search(r"WAIT_MOTION", next_ln, flags=re.I):
                        m_min = re.search(r"(\d+)\s*분", next_ln, flags=re.I)
                        if m_min:
                            wait_min = m_min.group(1)
                        j += 1
                    else:
                        break
                if rpm_val and wait_min:
                    merged = f"Shaker {rpm_val} rpm, {wait_min}분 배양."
                    print(f"[MERGE] {lines[i].strip()} + ... -> {merged}")
                    merged_lines.append(merged)
                    i = j
                    continue
            merged_lines.append(ln)
            i += 1
        lines = merged_lines

        # ── Preprocessing 2: merge split TRANS/TEMPPLATE_ON/WAIT_MOTION/TRANS blocks ──
        # When the model emits "TRANS 1→3 / TEMPPLATE_ON 20 / WAIT_MOTION 1800 / TRANS 3→1"
        # as individual lines -> merge into a single line "상온(벤치탑) 배양 N분"
        merged_lines2 = []
        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            # Detect the "TRANS X→3" (plate_slot -> hotplate_slot) pattern
            trans_start = re.match(r"^\s*-?\s*TRANS\s+(\d+)\s*[→\-\->]+\s*3\s*$", ln, flags=re.I)
            if trans_start and i + 3 < len(lines):
                ln1 = lines[i + 1].strip()
                ln2 = lines[i + 2].strip()
                ln3 = lines[i + 3].strip()
                # TEMPPLATE_ON N
                temp_m = re.match(r"^\s*-?\s*TEMPPLATE_ON\s+(\d+)\s*$", ln1, flags=re.I)
                # WAIT_MOTION (seconds or minutes)
                wait_m = re.match(r"^\s*-?\s*WAIT_MOTION\s+(.+)$", ln2, flags=re.I)
                # TRANS 3→X (hotplate_slot → plate_slot)
                trans_end = re.match(r"^\s*-?\s*TRANS\s+3\s*[→\-\->]+\s*(\d+)\s*$", ln3, flags=re.I)
                if temp_m and wait_m and trans_end:
                    # Extract the time from the WAIT_MOTION value
                    wait_str = wait_m.group(1).strip()
                    min_m = re.search(r"(\d+)\s*분", wait_str)
                    if min_m:
                        minutes = int(min_m.group(1))
                    else:
                        # If only a number is present, it is in seconds -> convert to minutes
                        sec_m = re.search(r"(\d+)", wait_str)
                        if sec_m:
                            minutes = max(int(sec_m.group(1)) // 60, 1)
                        else:
                            minutes = 30  # fallback
                    temp_val = int(temp_m.group(1))
                    if temp_val <= 25:
                        merged = f"상온(벤치탑) 배양 {minutes}분"
                    else:
                        merged = f"{temp_val}°C 배양 {minutes}분"
                    print(f"[MERGE] TRANS/TEMPPLATE/WAIT/TRANS -> {merged}")
                    merged_lines2.append(merged)
                    i += 4
                    continue
            merged_lines2.append(ln)
            i += 1
        lines = merged_lines2

        # ── Preprocessing 3: merge Row-dispense batches ──
        # Group consecutive Row-dispense lines from the same reservoir and handle them with a single tip
        # ELISA protocols have no Row pattern, so this preprocessing is a no-op for them
        # Pattern A: single Row - "Reservoir 1 Row 3에 45 µL 분주"
        _row_pat = re.compile(r"reservoir\s*(\d+).*?row\s*(\d+).*?(\d+)\s*[µu]?l.*?분주", flags=re.I)
        # Pattern B: Row range - "Reservoir 3 Row 1~8 전체 컬럼에 165 µL 분주"
        _row_range_pat = re.compile(
            r"reservoir\s*(\d+).*?row\s*(\d+)\s*[~\-]\s*(\d+).*?(\d+)\s*[µu]?l.*?분주", flags=re.I)
        remaining_lines = []
        i = 0
        while i < len(lines):
            # Check the range pattern first (Row N~M)
            mr = _row_range_pat.search(lines[i])
            if mr:
                rcol = int(mr.group(1))
                row_start = int(mr.group(2))
                row_end = int(mr.group(3))
                vol = int(mr.group(4))
                batch = [(r, vol) for r in range(row_start, row_end + 1)]
                print(f"[ROW-RANGE] Reservoir {rcol}: Row {row_start}~{row_end}, {vol} uL -> batch processing with a single tip")
                rows += build_row_dispense_batch(
                    allowed_cmds, slots, tipper, rcol, batch,
                    self.config.DEFAULTS["aspirate_speed"],
                    self.config.DEFAULTS["dispense_speed"])
                i += 1
                continue
            # Check the single Row pattern
            m = _row_pat.search(lines[i])
            if m:
                rcol = int(m.group(1))
                batch = [(int(m.group(2)), int(m.group(3)))]
                j = i + 1
                while j < len(lines):
                    next_m = _row_pat.search(lines[j])
                    if next_m and int(next_m.group(1)) == rcol:
                        batch.append((int(next_m.group(2)), int(next_m.group(3))))
                        j += 1
                    else:
                        break
                print(f"[ROW-BATCH] Reservoir {rcol}: {len(batch)} rows -> batch processing with a single tip")
                rows += build_row_dispense_batch(
                    allowed_cmds, slots, tipper, rcol, batch,
                    self.config.DEFAULTS["aspirate_speed"],
                    self.config.DEFAULTS["dispense_speed"])
                i = j
                continue
            remaining_lines.append(lines[i])
            i += 1
        lines = remaining_lines

        # Process all lines of the protocol in order
        processed_lines = []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith('[') or ln.startswith('#'):
                continue
            # Skip prompt hint lines (starting with ★ or (★)
            if ln.startswith('★') or ln.startswith('(★'):
                print(f"[SKIP] Prompt hint line: {ln}")
                continue

            # Debug logging
            print(f"[PROCESSING] {ln}")

            seq = line_to_sequence(ln, allowed_cmds, slots, tipper, self.config.DEFAULTS, self.wellplate_coords)

            if seq:
                has_not_mapped = any(cmd.get("Command") == "NOT_MAPPED" for cmd in seq)
                if has_not_mapped:
                    print(f"[WARNING] Mapping failed: {ln}")
                    # Record NOT_MAPPED as well (for debugging)
                    processed_lines.append({"line": ln, "status": "NOT_MAPPED"})
                else:
                    rows += seq
                    processed_lines.append({"line": ln, "status": "OK", "commands": len(seq)})
            else:
                print(f"[SKIP] Empty sequence: {ln}")

        # Processing result summary
        print(f"\n[SUMMARY] {len(processed_lines)} lines processed in total")
        print(f"  - Success: {sum(1 for p in processed_lines if p.get('status') == 'OK')}")
        print(f"  - Failed: {sum(1 for p in processed_lines if p.get('status') == 'NOT_MAPPED')}")

        # OD measurement handling (section 6)
        od_pattern = r"(?:\bO\.?\s*D\.?\b|OD측정|흡광도?|Absorbance|450\s*nm|620\s*nm)"
        if re.search(od_pattern, section6, flags=re.I):
            rows += build_elisa_readout_block(allowed_cmds, slots)
        else:
            # When there is no OD readout: termination block (close cap -> store plate)
            add(rows, allowed_cmds, 73, "A#CAP_CLOSE", "1 9")
            add(rows, allowed_cmds, 74, "A#GET", "4 1")
            add(rows, allowed_cmds, 74, "A#PUT", "2 1")

        # Save the file
        txt_name = txt_path.name
        prefix = Path(txt_name).stem.split("_")[0]
        out_path = self.config.SEQUENCE_OUTPUT_DIR / f"{prefix}{self.config.SEQUENCE_SUFFIX}"

        df = pd.DataFrame(rows, columns=["Handler", "Command", "Input Parameters"])
        df.to_excel(out_path, index=False)
        print(f"Done: {out_path} ({len(df)} rows in total)")