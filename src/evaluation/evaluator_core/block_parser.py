"""
block_parser.py — Split a sequence xlsx into logical blocks.
Group consecutive commands into blocks such as incubation/wash/dispense/
solution-removal/shaking/readout, producing comparable units.
"""

import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SeqBlock:
    """A single logical block"""
    type: str                        # initial-setup|incubation|solution-removal|wash|dispense|shaking|readout
    start_idx: int                   # start row index (0-based)
    end_idx: int                     # end row index (inclusive)
    reservoir_col: Optional[int] = None
    wait_seconds: Optional[int] = None
    temperature: Optional[int] = None
    volume: Optional[int] = None     # representative volume (first ASPIRATE/DISPENSE)
    wash_cycles: Optional[int] = None
    rpm: Optional[int] = None
    shake_seconds: Optional[int] = None
    _df: object = field(default=None, repr=False)  # DataFrame of rows in the block (for row-level comparison)

    def match_key(self) -> Tuple:
        """LCS matching key — type only. Reservoir excluded for all types
        because Wash Buffer position change can cascade-shift all reservoir numbers.
        Reservoir differences are caught as parameter errors instead."""
        return (self.type,)

    def param_dict(self) -> dict:
        """Return comparable parameters.
        Reservoir excluded for all types — Wash Buffer position change
        can cascade-shift all reservoir numbers. Reservoir mapping is
        evaluated separately by reservoir_evaluator_llm."""
        d = {}
        if self.wait_seconds is not None:
            d["wait_seconds"] = self.wait_seconds
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.volume is not None:
            d["volume"] = self.volume
        if self.wash_cycles is not None:
            d["wash_cycles"] = self.wash_cycles
        # reservoir_col excluded — evaluated separately in reservoir mapping
        if self.rpm is not None:
            d["rpm"] = self.rpm
        if self.shake_seconds is not None:
            d["shake_seconds"] = self.shake_seconds
        return d

    def short_desc(self) -> str:
        parts = [self.type]
        if self.reservoir_col is not None:
            parts.append(f"R{self.reservoir_col}")
        if self.volume is not None:
            parts.append(f"{self.volume}µL")
        if self.wait_seconds is not None:
            parts.append(f"{self.wait_seconds}s")
        if self.wash_cycles is not None:
            parts.append(f"{self.wash_cycles}회")
        if self.rpm is not None:
            parts.append(f"{self.rpm}rpm")
        return " ".join(parts)


def _parse_params(param_str) -> List[int]:
    """Parse the Input Parameters string into a list of integers"""
    if pd.isna(param_str) or str(param_str).strip() == "":
        return []
    parts = str(param_str).strip().split()
    result = []
    for p in parts:
        try:
            result.append(int(float(p)))
        except ValueError:
            pass
    return result


def _get_cmd(row) -> str:
    return str(row.get("Command", "")).strip()


def _get_handler(row) -> int:
    try:
        return int(row.get("Handler", 0))
    except (ValueError, TypeError):
        return 0


def _get_params(row) -> List[int]:
    return _parse_params(row.get("Input Parameters", ""))


def parse_blocks(df: pd.DataFrame, max_rows: int = 2000) -> List[SeqBlock]:
    """
    Scan the sequence DataFrame sequentially and convert it into a list of logical blocks.

    Block identification strategy:
    - TRANS x 3 -> start of an incubation block (Deck 3 = incubator)
    - TRANS x 6 -> start of a shaking block (Deck 6 = shaker)
    - ASPIRATE_FROM_PLATE + DISPENSE_INTO_PLATE patterns distinguish wash/dispense/removal
    - ANALYZER family -> readout block
    - Leading part (before the first TRANS/GET/PUT) -> initial setup

    If max_rows is exceeded, truncate before parsing (performance guard).
    """
    import time as _t
    _parse_start = _t.time()
    _PARSE_TIMEOUT = 10  # at most 10 seconds

    if len(df) > max_rows:
        print(f"    [parse_blocks] rows {len(df)} > {max_rows} - truncating to {max_rows}", flush=True)
        df = df.head(max_rows)

    rows = df.to_dict("records")
    n = len(rows)
    blocks = []
    i = 0
    _ppb_cache = {}  # cache of _parse_pipette_block results (start_idx -> result)

    # Initial-setup block (up to the first incubation/dispense/wash)
    init_end = _find_first_main_action(rows)
    if init_end > 0:
        blocks.append(SeqBlock(type="초기설정", start_idx=0, end_idx=init_end - 1))  # initial setup
        i = init_end

    while i < n:
        if _t.time() - _parse_start > _PARSE_TIMEOUT:
            print(f"    [parse_blocks] timeout ({_PARSE_TIMEOUT}s) - {len(blocks)} blocks parsed", flush=True)
            break
        cmd = _get_cmd(rows[i])
        params = _get_params(rows[i])

        # ── Incubation block: TRANS -> Deck 3 ──
        if cmd == "A#TRANS" and len(params) >= 2 and params[1] == 3:
            block_start = i
            temp = None
            wait_sec = None
            # Scan: from TRANS->3 through TEMPPLATE_ON, WAIT_MOTION, up to TRANS 3->
            j = i + 1
            while j < n:
                jcmd = _get_cmd(rows[j])
                jparams = _get_params(rows[j])
                if jcmd == "A#TEMPPLATE_ON":
                    temp = jparams[0] if jparams else None
                elif jcmd == "A#WAIT_MOTION":
                    wait_sec = jparams[0] if jparams else None
                elif jcmd == "A#TRANS" and len(jparams) >= 2 and jparams[0] == 3:
                    # TRANS 3->x = end of incubation block
                    blocks.append(SeqBlock(
                        type="배양", start_idx=block_start, end_idx=j,
                        temperature=temp, wait_seconds=wait_sec,
                    ))
                    i = j + 1
                    break
                j += 1
            else:
                # TRANS 3-> not found — incomplete incubation block
                blocks.append(SeqBlock(
                    type="배양", start_idx=block_start, end_idx=min(j, n - 1),
                    temperature=temp, wait_seconds=wait_sec,
                ))
                i = j
            continue

        # ── Shaking block: TRANS -> Deck 6 ──
        if cmd == "A#TRANS" and len(params) >= 2 and params[1] == 6:
            block_start = i
            rpm = None
            shake_sec = None
            j = i + 1
            while j < n:
                jcmd = _get_cmd(rows[j])
                jparams = _get_params(rows[j])
                if jcmd == "A#SHAKE_START" and len(jparams) >= 2:
                    rpm = jparams[0]
                    shake_sec = jparams[1]
                elif jcmd == "A#TRANS" and len(jparams) >= 2 and jparams[0] == 6:
                    blocks.append(SeqBlock(
                        type="shaking", start_idx=block_start, end_idx=j,
                        rpm=rpm, shake_seconds=shake_sec,
                    ))
                    i = j + 1
                    break
                j += 1
            else:
                blocks.append(SeqBlock(
                    type="shaking", start_idx=block_start, end_idx=min(j, n - 1),
                    rpm=rpm, shake_seconds=shake_sec,
                ))
                i = j
            continue

        # ── Shaking block (when SHAKE_START appears directly, without TRANS) ──
        if cmd == "A#SHAKE_START":
            sparams = _get_params(rows[i])
            rpm = sparams[0] if len(sparams) >= 1 else None
            shake_sec = sparams[1] if len(sparams) >= 2 else None
            blocks.append(SeqBlock(
                type="shaking", start_idx=i, end_idx=i,
                rpm=rpm, shake_seconds=shake_sec,
            ))
            i += 1
            continue

        # ── Readout block: ANALYZER series ──
        if cmd in ("A#ANALYZER_OPEN", "A#ANALYZER_START", "A#ANALYZER_CLOSE"):
            block_start = i
            j = i + 1
            while j < n:
                jcmd = _get_cmd(rows[j])
                if jcmd.startswith("A#ANALYZER") or jcmd in ("A#TRANS", "A#CAP_OPEN", "A#CAP_CLOSE"):
                    j += 1
                else:
                    break
            blocks.append(SeqBlock(type="판독", start_idx=block_start, end_idx=j - 1))
            i = j
            continue

        # ── Pipette block: SELECT_CHANNEL -> INSTALL -> action -> EJECT ──
        if cmd in ("A#ADP_SELECT_4_CHANNEL", "A#ADP_SELECT_1_CHANNEL"):
            block_start = i
            if i in _ppb_cache:
                block_type, reservoir_col, volume, wash_cycles = _ppb_cache[i]
            else:
                block_type, reservoir_col, volume, wash_cycles = _parse_pipette_block(rows, i, n)
                _ppb_cache[i] = (block_type, reservoir_col, volume, wash_cycles)
            # Scan up to EJECT
            j = i + 1
            eject_found = False
            _merge_limit = 50  # maximum number of consecutive blocks to merge (prevents infinite loop)
            _merge_count = 0
            while j < n:
                jcmd = _get_cmd(rows[j])
                if jcmd == "A#ADP_EJECT_PIPETTE":
                    eject_found = True
                    j += 1
                    # peek: check whether the next SELECT continues the same logical action
                    if _merge_count < _merge_limit and j < n and _get_cmd(rows[j]) in ("A#ADP_SELECT_4_CHANNEL", "A#ADP_SELECT_1_CHANNEL"):
                        if j in _ppb_cache:
                            next_bt, next_rc, next_vol, next_wc = _ppb_cache[j]
                        else:
                            next_bt, next_rc, next_vol, next_wc = _parse_pipette_block(rows, j, n)
                            _ppb_cache[j] = (next_bt, next_rc, next_vol, next_wc)
                        if next_bt == block_type and next_rc == reservoir_col:
                            _merge_count += 1
                            continue
                    # End of block
                    blocks.append(SeqBlock(
                        type=block_type, start_idx=block_start, end_idx=j - 1,
                        reservoir_col=reservoir_col, volume=volume,
                        wash_cycles=wash_cycles,
                    ))
                    i = j
                    break
                j += 1
            if not eject_found:
                blocks.append(SeqBlock(
                    type=block_type, start_idx=block_start, end_idx=min(j, n - 1),
                    reservoir_col=reservoir_col, volume=volume,
                    wash_cycles=wash_cycles,
                ))
                i = j
            continue

        # ── CAP_OPEN / CAP_CLOSE / GET / PUT — readout start or standalone ──
        if cmd in ("A#CAP_CLOSE", "A#CAP_OPEN"):
            # May be part of a readout block — check whether GET/PUT/ANALYZER follows
            j = i + 1
            if j < n and _get_cmd(rows[j]) in ("A#GET", "A#PUT", "A#ANALYZER_OPEN"):
                # Treat as the start of a readout block
                block_start = i
                while j < n:
                    jcmd = _get_cmd(rows[j])
                    if jcmd in ("A#GET", "A#PUT", "A#ANALYZER_OPEN", "A#ANALYZER_CLOSE",
                                "A#ANALYZER_START", "A#CAP_OPEN", "A#CAP_CLOSE", "A#TRANS"):
                        j += 1
                    else:
                        break
                blocks.append(SeqBlock(type="판독", start_idx=block_start, end_idx=j - 1))
                i = j
                continue

        # Skip (unclassifiable row)
        i += 1

    # Store the corresponding row DataFrame slice in each block (for row-level comparison)
    for b in blocks:
        b._df = df.iloc[b.start_idx:b.end_idx + 1].reset_index(drop=True)

    return blocks


def _find_first_main_action(rows) -> int:
    """Find the start index of the first incubation/dispense/wash block"""
    for i, row in enumerate(rows):
        cmd = _get_cmd(row)
        params = _get_params(row)
        # TRANS ->3 (incubation), ->6 (shaking), or SELECT_CHANNEL (pipette)
        if cmd == "A#TRANS" and len(params) >= 2 and params[1] in (3, 6):
            return i
        if cmd in ("A#ADP_SELECT_4_CHANNEL", "A#ADP_SELECT_1_CHANNEL"):
            return i
    return len(rows)


def _parse_pipette_block(rows, start_idx, n) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """
    Determine the pipette block type starting from a SELECT_CHANNEL point.
    Returns: (block_type, reservoir_col, volume, wash_cycles)

    wash: repeated ASPIRATE(plate) -> DISPENSE(plate) + reservoir=1 source
    dispense: ASPIRATE(reservoir, slot=5) -> DISPENSE(plate, slot=1)
    solution-removal: ASPIRATE(plate, slot=1) -> DISPENSE(waste, slot=8)
    """
    reservoir_col = None
    volume = None
    aspirate_from_plate = 0
    aspirate_from_reservoir = 0
    dispense_to_plate = 0
    dispense_to_waste = 0
    wash_cycle_count = 0

    j = start_idx + 1
    _scan_limit = min(n, start_idx + 200)  # scan at most 200 rows (performance guard)
    while j < _scan_limit:
        cmd = _get_cmd(rows[j])
        params = _get_params(rows[j])

        if cmd == "A#ADP_EJECT_PIPETTE":
            break

        if cmd == "A#ADP_ASPIRATE_FROM_PLATE" and len(params) >= 4:
            slot = params[0]
            vol = params[3]
            if volume is None:
                volume = vol
            if slot == 5:
                # Aspirate from reservoir (slot 5)
                aspirate_from_reservoir += 1
                col = params[1]
                reservoir_col = col
            else:
                aspirate_from_plate += 1

        elif cmd == "A#ADP_DISPENSE_INTO_PLATE" and len(params) >= 4:
            slot = params[0]
            vol = params[3]
            if volume is None:
                volume = vol
            if slot == 8:
                dispense_to_waste += 1
            elif slot == 1:
                dispense_to_plate += 1

        # If the next SELECT_CHANNEL appears, it is a channel switch within the same logical block
        if cmd in ("A#ADP_SELECT_4_CHANNEL", "A#ADP_SELECT_1_CHANNEL") and j > start_idx + 1:
            break

        j += 1

    # Determine the type
    if aspirate_from_reservoir > 0 and dispense_to_plate > 0 and aspirate_from_plate > 0:
        # aspirate from reservoir + dispense to plate + aspirate from plate = wash pattern
        wash_cycle_count = dispense_to_plate
        return ("세척", reservoir_col, volume, wash_cycle_count)
    elif aspirate_from_reservoir > 0 and dispense_to_plate > 0:
        return ("분주", reservoir_col, volume, None)
    elif aspirate_from_plate > 0 and dispense_to_waste > 0:
        return ("용액제거", None, volume, None)
    elif aspirate_from_plate > 0 and dispense_to_plate > 0:
        # plate->plate = wash (aspirate old -> dispense wash)
        return ("세척", reservoir_col, volume, dispense_to_plate)
    else:
        return ("분주", reservoir_col, volume, None)
