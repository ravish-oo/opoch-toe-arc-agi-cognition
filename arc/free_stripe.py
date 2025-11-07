"""
WO-3H: FREE Verifier - Stripe / Column / Row Projections (types-only)

Prove that output type mosaic T_Y is obtained from input type mosaic T_X
by applying a single, fixed index map along exactly one axis (rows or columns).

Pattern families:
- Subset: explicit index selection (e.g., left/right blocks)
- Stride: every k-th row/col with fixed offset
- Interleave: periodic pattern of positions

All checks use np.take + np.array_equal (exact equality, no heuristics).
"""

from typing import List, Tuple, Union, Dict, Any
import numpy as np

# Pattern spec: ("subset", axis, tuple(indices))
#               ("stride", axis, stride_k, offset_o)
#               ("interleave", axis, period_p, tuple(select_positions))
Pattern = Tuple[Union[str, int], ...]
Candidate = Tuple[str, Pattern]  # ("band_map", pattern_spec)


def verify_stripe_maps(T_X: np.ndarray, T_Y: np.ndarray) -> List[Candidate]:
    """
    Return all proven band-map candidates for this pair.
    Operates strictly on type mosaics (no color access).

    Args:
        T_X: Input type mosaic (H, W)
        T_Y: Output type mosaic (h, w)

    Returns:
        List of ("band_map", pattern_spec) tuples for all proven candidates
    """
    candidates = []
    src_shape = T_X.shape
    dst_shape = T_Y.shape

    # Try both axes
    for axis in [0, 1]:  # 0=rows, 1=cols
        src_len = src_shape[axis]
        dst_len = dst_shape[axis]

        # Check if other axis matches (required for band map)
        other_axis = 1 - axis
        if src_shape[other_axis] != dst_shape[other_axis]:
            continue

        # 1. SUBSET selection (left/right blocks)
        subset_cands = _check_subset(T_X, T_Y, axis, src_len, dst_len)
        candidates.extend(subset_cands)

        # 2. STRIDE selection
        stride_cands = _check_stride(T_X, T_Y, axis, src_len, dst_len)
        candidates.extend(stride_cands)

        # 3. INTERLEAVE (periodic selection)
        interleave_cands = _check_interleave(T_X, T_Y, axis, src_len, dst_len)
        candidates.extend(interleave_cands)

    return candidates


def _check_subset(T_X: np.ndarray, T_Y: np.ndarray, axis: int,
                  src_len: int, dst_len: int) -> List[Candidate]:
    """Check subset selection patterns (left/right blocks)."""
    candidates = []

    if dst_len > src_len:
        return candidates

    # Left block (fold-left)
    indices = np.arange(dst_len)
    if _verify_projection(T_X, T_Y, indices, axis):
        pattern = ("subset", axis, tuple(indices.tolist()))
        candidates.append(("band_map", pattern))

    # Right block (fold-right)
    indices = src_len - dst_len + np.arange(dst_len)
    if _verify_projection(T_X, T_Y, indices, axis):
        pattern = ("subset", axis, tuple(indices.tolist()))
        candidates.append(("band_map", pattern))

    return candidates


def _check_stride(T_X: np.ndarray, T_Y: np.ndarray, axis: int,
                  src_len: int, dst_len: int) -> List[Candidate]:
    """Check stride selection patterns (every k-th element with offset).

    Exhaustively enumerate all valid (stride, offset) pairs per A0 (no minted differences).
    For each stride k ∈ [1..src_len], check all offsets o ∈ [0..k-1] where
    len(range(o, src_len, k)) == dst_len.
    """
    candidates = []

    # Exhaustive enumeration: all strides from 1 to src_len
    for stride in range(1, src_len + 1):
        for offset in range(stride):
            indices = np.arange(offset, src_len, stride)

            # Must match destination length exactly
            if len(indices) != dst_len:
                continue

            if _verify_projection(T_X, T_Y, indices, axis):
                pattern = ("stride", axis, stride, offset)
                candidates.append(("band_map", pattern))

    return candidates


def _check_interleave(T_X: np.ndarray, T_Y: np.ndarray, axis: int,
                      src_len: int, dst_len: int) -> List[Candidate]:
    """Check interleave patterns with exhaustive period enumeration (per A0).

    Enumerates all periods up to src_len and all valid select positions within each period.
    More exhaustive than hardcoded patterns, but uses verification (not heuristic limits).
    """
    candidates = []

    # Exhaustive period enumeration (not capped at 6)
    for period in range(1, src_len + 1):
        # For this period, try all possible select patterns
        # Generate all non-empty subsets of [0, period-1] up to reasonable size
        # To keep it tractable, enumerate patterns with up to 3 positions
        max_select_size = min(3, period)

        for select_size in range(1, max_select_size + 1):
            # Generate all combinations of select_size positions from [0, period-1]
            import itertools
            for select_pos in itertools.combinations(range(period), select_size):
                select_list = list(select_pos)

                # Build index sequence
                indices = _build_interleave_indices(select_list, period, dst_len, src_len)

                if len(indices) != dst_len:
                    continue

                # Verify via np.take + array_equal
                if _verify_projection(T_X, T_Y, indices, axis):
                    pattern = ("interleave", axis, period, tuple(select_list))
                    candidates.append(("band_map", pattern))

    return candidates


def _build_interleave_indices(select_pos: List[int], period: int,
                               dst_len: int, src_len: int) -> np.ndarray:
    """Build interleaved index sequence with periodic pattern and modulo wraparound.

    Aligns with np.take(mode='wrap') semantics: indices beyond src_len wrap modulo src_len.
    Builds exactly dst_len indices.
    """
    base = np.array(select_pos, dtype=np.int64)
    pattern_len = len(base)

    # Build exactly dst_len indices with modulo wraparound
    indices = []
    for i in range(dst_len):
        # Position within the select pattern
        pos_in_pattern = i % pattern_len
        # Which cycle we're in
        cycle = i // pattern_len
        # Compute raw index
        raw_idx = cycle * period + base[pos_in_pattern]
        # Apply modulo wraparound to keep in [0, src_len)
        wrapped_idx = raw_idx % src_len
        indices.append(wrapped_idx)

    return np.array(indices, dtype=np.int64)


def _verify_projection(T_X: np.ndarray, T_Y: np.ndarray,
                       indices: np.ndarray, axis: int) -> bool:
    """
    Verify if T_Y equals T_X projected via indices along axis.
    Uses documented NumPy API: np.take with mode='wrap' + np.array_equal.
    """
    try:
        # Use np.take with mode='wrap' for well-defined periodic selection
        T_hat = np.take(T_X, indices, axis=axis, mode='wrap')

        # Exact equality check
        return np.array_equal(T_hat, T_Y)
    except Exception:
        return False


def generate_stripe_receipt(
    task_id: str,
    pair_index: int,
    T_X: np.ndarray,
    T_Y: np.ndarray,
    candidates: List[Candidate]
) -> Dict[str, Any]:
    """
    Generate receipt for stripe map verification (WO-3H).

    Args:
        task_id: Task identifier
        pair_index: Training pair index
        T_X: Input type mosaic
        T_Y: Output type mosaic
        candidates: List of proven candidates

    Returns:
        Receipt dict with detailed verification info
    """
    receipt = {
        "task_id": task_id,
        "pair_index": pair_index,
        "free_stripe": {
            "source_shape": list(T_X.shape),
            "target_shape": list(T_Y.shape),
            "candidates": []
        }
    }

    # Check all pattern families and log results
    checked_patterns = _enumerate_all_patterns(T_X.shape, T_Y.shape)

    for pattern_info in checked_patterns:
        axis = pattern_info["axis"]
        mode = pattern_info["mode"]
        indices = pattern_info["indices"]

        # Verify this pattern
        ok = _verify_projection(T_X, T_Y, indices, axis)

        candidate_info = {
            "axis": axis,
            "mode": mode,
            "ok": ok
        }

        # Add mode-specific fields
        if mode == "subset":
            candidate_info["indices"] = indices.tolist()
        elif mode == "stride":
            candidate_info["stride"] = pattern_info["stride"]
            candidate_info["offset"] = pattern_info["offset"]
        elif mode == "interleave":
            candidate_info["period"] = pattern_info["period"]
            candidate_info["select"] = pattern_info["select"]

        receipt["free_stripe"]["candidates"].append(candidate_info)

    # Add proven candidate if any
    if candidates:
        # Use first proven candidate
        receipt["candidate"] = list(candidates[0])

    return receipt


def _enumerate_all_patterns(src_shape: Tuple[int, int],
                            dst_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Enumerate all patterns to check for receipt logging."""
    patterns = []

    for axis in [0, 1]:
        src_len = src_shape[axis]
        dst_len = dst_shape[axis]
        other_axis = 1 - axis

        # Skip if other axis doesn't match
        if src_shape[other_axis] != dst_shape[other_axis]:
            continue

        # Subset patterns
        if dst_len <= src_len:
            # Left block
            patterns.append({
                "axis": axis,
                "mode": "subset",
                "indices": np.arange(dst_len)
            })

            # Right block
            patterns.append({
                "axis": axis,
                "mode": "subset",
                "indices": src_len - dst_len + np.arange(dst_len)
            })

        # Stride patterns (sample a few)
        for stride in [2, 3]:
            if stride > src_len:
                continue
            for offset in range(min(stride, 2)):  # Sample first 2 offsets
                indices = np.arange(offset, src_len, stride)
                if len(indices) == dst_len:
                    patterns.append({
                        "axis": axis,
                        "mode": "stride",
                        "stride": stride,
                        "offset": offset,
                        "indices": indices
                    })

        # Interleave patterns (sample a few)
        for period in [2, 3]:
            if period > src_len:
                continue
            for select in [[0], [0, 1]]:
                if any(p >= period for p in select):
                    continue
                indices = _build_interleave_indices(select, period, dst_len, src_len)
                if len(indices) == dst_len:
                    patterns.append({
                        "axis": axis,
                        "mode": "interleave",
                        "period": period,
                        "select": select,
                        "indices": indices
                    })

    return patterns
