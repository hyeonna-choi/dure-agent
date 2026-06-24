"""
matcher.py — LCS-based matching algorithm.
Matches a list of Actions or a list of SeqBlocks while preserving order.
"""

from typing import List, Tuple, Any


def _items_match(a, b) -> bool:
    """Two items are match candidates if their match_key values are equal"""
    return a.match_key() == b.match_key()


def lcs_match(gt_list: List, pred_list: List, max_dp_cells: int = 500_000) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    LCS (Longest Common Subsequence) based matching.
    Falls back to greedy matching when n*m > max_dp_cells (performance guard).

    Args:
        gt_list: list of ground-truth items (Action or SeqBlock)
        pred_list: list of predicted items
        max_dp_cells: maximum DP table size (greedy fallback when exceeded)

    Returns:
        matched_pairs: [(gt_idx, pred_idx), ...] — TP pairs
        unmatched_gt: [gt_idx, ...] — FN (present only in ground truth)
        unmatched_pred: [pred_idx, ...] — FP (present only in prediction)
    """
    n = len(gt_list)
    m = len(pred_list)

    # Too many blocks -> greedy matching (O(n*m) DP not feasible)
    if n * m > max_dp_cells:
        print(f"    [lcs_match] DP table {n}x{m}={n*m} > {max_dp_cells} -> greedy fallback", flush=True)
        return _greedy_match(gt_list, pred_list)

    # LCS DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if _items_match(gt_list[i - 1], pred_list[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack -> extract matched pairs
    matched_pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if _items_match(gt_list[i - 1], pred_list[j - 1]):
            matched_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    matched_pairs.reverse()

    # Indices matched as TP
    matched_gt = {p[0] for p in matched_pairs}
    matched_pred = {p[1] for p in matched_pairs}

    # FN / FP
    unmatched_gt = [i for i in range(n) if i not in matched_gt]
    unmatched_pred = [j for j in range(m) if j not in matched_pred]

    return matched_pairs, unmatched_gt, unmatched_pred


def _greedy_match(gt_list: List, pred_list: List) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy sequential matching used when O(n*m) DP is not feasible.
    Scans pred from the start and matches against gt in order.
    """
    n = len(gt_list)
    m = len(pred_list)
    matched_pairs = []
    pred_cursor = 0

    for gi in range(n):
        for pj in range(pred_cursor, m):
            if _items_match(gt_list[gi], pred_list[pj]):
                matched_pairs.append((gi, pj))
                pred_cursor = pj + 1
                break

    matched_gt = {p[0] for p in matched_pairs}
    matched_pred = {p[1] for p in matched_pairs}
    unmatched_gt = [i for i in range(n) if i not in matched_gt]
    unmatched_pred = [j for j in range(m) if j not in matched_pred]

    return matched_pairs, unmatched_gt, unmatched_pred


def compare_params(gt_item, pred_item) -> Tuple[int, int, List[dict]]:
    """
    Compare the parameters of a matched pair.
    For SeqBlock items, row-count coverage is also compared as a parameter.

    Returns:
        correct: number of matching parameters
        total: total number of comparable parameters
        details: [{param, gt_val, pred_val, match}, ...]
    """
    gt_params = gt_item.param_dict()
    pred_params = pred_item.param_dict()

    # Union comparison: any key present in GT or Pred is compared
    all_keys = set(gt_params.keys()) | set(pred_params.keys())

    correct = 0
    total = 0
    details = []

    # Check whether the type is shaking (allow default values for short shaking)
    _is_shaking = False
    if hasattr(gt_item, 'type'):
        _is_shaking = (gt_item.type == "shaking")

    for key in sorted(all_keys):
        gt_val = gt_params.get(key)
        pred_val = pred_params.get(key)
        total += 1
        match = (gt_val == pred_val)
        # shaking type: if GT is None (not specified), any Pred value is allowed
        # (sequence_builder inserts default values 200rpm/10s, so it is valid even if GPT writes them as text)
        if _is_shaking and gt_val is None and pred_val is not None:
            match = True
        if _is_shaking and gt_val is not None and pred_val is None:
            match = True
        if match:
            correct += 1
        details.append({
            "param": key,
            "gt_val": gt_val,
            "pred_val": pred_val,
            "match": match,
        })

    # SeqBlock: row-level comparison (unweighted, actual row matching)
    # Greedy-match each row of the GT block against each row of the Pred block by Command
    if hasattr(gt_item, 'start_idx') and hasattr(gt_item, 'end_idx') and hasattr(gt_item, '_df'):
        gt_rows_data = gt_item._df
        pred_rows_data = pred_item._df
        if gt_rows_data is not None and pred_rows_data is not None and len(gt_rows_data) > 0:
            row_correct, row_total = _compare_rows(gt_rows_data, pred_rows_data)
            correct += row_correct
            total += row_total
            details.append({
                "param": "row_commands",
                "gt_val": len(gt_rows_data),
                "pred_val": len(pred_rows_data),
                "match": row_correct == row_total,
                "row_correct": row_correct,
                "row_total": row_total,
            })

    return correct, total, details


def _normalize_params_for_compare(cmd: str, params: str, reservoir_map: dict = None) -> str:
    """
    Normalize Input Parameters before comparison.
    - Allow Wash Buffer reservoir swap: normalize the reservoir column number
    - For ASPIRATE/DISPENSE family commands, if the second number (reservoir col)
      is in reservoir_map, replace it with the normalized value
    """
    if not reservoir_map:
        return params
    # Normalize only ASPIRATE/DISPENSE family commands (reservoir col is the second parameter)
    aspirate_dispense = ("ASPIRATE", "DISPENSE")
    if not any(kw in cmd.upper() for kw in aspirate_dispense):
        return params
    parts = params.split()
    if len(parts) >= 2:
        col_str = parts[1]
        if col_str in reservoir_map:
            parts[1] = reservoir_map[col_str]
            return " ".join(parts)
    return params


def _build_reservoir_swap_map(gt_df, pred_df) -> dict:
    """
    When a common reagent's reservoir column is swapped between the GT and Pred
    sequences, build a Pred column -> GT column mapping.

    Collect patterns where, at the same row position, the Command and the remaining
    parameters are equal but only the second parameter (col) differs, and build a
    consistent swap mapping.
    """
    from collections import Counter

    # Compare rows at the same index to collect col swap patterns
    swap_counts = Counter()  # (pred_col, gt_col) -> count

    for i in range(min(len(gt_df), len(pred_df))):
        gt_cmd = str(gt_df.iloc[i].get("Command", ""))
        gt_params = str(gt_df.iloc[i].get("Input Parameters", ""))
        pred_cmd = str(pred_df.iloc[i].get("Command", ""))
        pred_params = str(pred_df.iloc[i].get("Input Parameters", ""))

        # Target only ASPIRATE/DISPENSE
        if gt_cmd != pred_cmd:
            continue
        if not any(kw in gt_cmd.upper() for kw in ("ASPIRATE", "DISPENSE")):
            continue

        gt_parts = gt_params.split()
        pred_parts = pred_params.split()
        if len(gt_parts) < 2 or len(pred_parts) < 2:
            continue

        # Case where only col (the second) differs and the rest is the same
        gt_col, pred_col = gt_parts[1], pred_parts[1]
        gt_rest = gt_parts[0] + "|" + "|".join(gt_parts[2:])
        pred_rest = pred_parts[0] + "|" + "|".join(pred_parts[2:])

        if gt_col != pred_col and gt_rest == pred_rest:
            swap_counts[(pred_col, gt_col)] += 1

    # Map only consistent swaps (appearing at least twice)
    swap_map = {}
    for (pc, gc), count in swap_counts.items():
        if count >= 2:
            swap_map[pc] = gc

    return swap_map


def _compare_rows(gt_df, pred_df) -> Tuple[int, int]:
    """
    Row-level comparison within a block. Greedy-match GT rows and Pred rows by
    Command+Parameters. Allow Wash Buffer reservoir swap: ignore reservoir column differences.
    Returns: (correct, total)
      - total = number of GT rows (denominator = all GT rows)
      - correct = number of matched rows whose parameters also match
    """
    # Build the reservoir swap mapping
    swap_map = _build_reservoir_swap_map(gt_df, pred_df)

    gt_rows = [(str(r.get("Command", "")),
                _normalize_params_for_compare(str(r.get("Command", "")),
                                              str(r.get("Input Parameters", "")),
                                              {}))  # GT needs no normalization
               for _, r in gt_df.iterrows()]
    pred_rows = [(str(r.get("Command", "")),
                  _normalize_params_for_compare(str(r.get("Command", "")),
                                                str(r.get("Input Parameters", "")),
                                                swap_map))  # Replace Pred col with GT col
                 for _, r in pred_df.iterrows()]

    total = len(gt_rows)
    if total == 0:
        return 0, 0

    # Greedy matching: for GT rows in order, find the same (Command, Params) in Pred
    correct = 0
    pred_cursor = 0
    for gt_cmd, gt_params in gt_rows:
        for pj in range(pred_cursor, len(pred_rows)):
            p_cmd, p_params = pred_rows[pj]
            if gt_cmd == p_cmd and gt_params == p_params:
                correct += 1
                pred_cursor = pj + 1
                break

    return correct, total


def check_order_match(gt_list: List, pred_list: List, matched_pairs: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Check whether the order of matched blocks agrees with the ground truth.

    Returns:
        correct_order: number of blocks in the correct order
        total_matched: total number of matched blocks
    """
    if len(matched_pairs) <= 1:
        return len(matched_pairs), len(matched_pairs)

    correct = 1  # The first one is always considered correct
    for k in range(1, len(matched_pairs)):
        prev_gt, prev_pred = matched_pairs[k - 1]
        curr_gt, curr_pred = matched_pairs[k]
        # Whether the order in GT matches the order in Pred
        if (curr_gt > prev_gt) == (curr_pred > prev_pred):
            correct += 1

    return correct, len(matched_pairs)
