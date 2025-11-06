"""
WO-3A & WO-3B: FREE Verifiers

WO-3A - Simple verifiers at color level:
  - identity (shape equality)
  - h-mirror-concat (2 variants)
  - v-double (vertical duplicate)
  - h-concat-dup (horizontal duplicate)
  - v-concat-dup (vertical duplicate)

WO-3B - Types-periodic tile verifier:
  - tile (integer blow-up on types, not colors)

All verifiers use exact NumPy equality checks (no approximations).
"""

from typing import List, Tuple, Optional, Any

import numpy as np


# Type aliases per WO-3A spec
Kind = str
Params = Optional[Tuple[Any, ...]]


def verify_simple_free(X: np.ndarray, Y: np.ndarray) -> List[Tuple[Kind, Params]]:
    """
    Return all simple FREE candidates verified exactly for this (X->Y) pair.

    Checks five simple verifiers at color level (no types):
      1. identity: shape equality (h==H, w==W)
      2. h-mirror-concat: [fliplr(X)|X] or [X|fliplr(X)]
      3. v-double: [X; X] (vertical duplicate)
      4. h-concat-dup: [X|X] (horizontal duplicate)
      5. v-concat-dup: [X;X] (vertical duplicate)

    Candidates are returned in deterministic order:
      identity, h-mirror-concat(rev|id), h-mirror-concat(id|rev),
      v-double, h-concat-dup, v-concat-dup

    Args:
        X: Input grid (H, W)
        Y: Output grid (h, w)

    Returns:
        List of (kind, params) tuples for all matching verifiers
    """
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)

    H, W = X.shape
    h, w = Y.shape

    candidates: List[Tuple[Kind, Params]] = []

    # 1. Identity: shape equality only
    if h == H and w == W:
        candidates.append(("identity", None))

    # 2. h-mirror-concat: two symmetric variants
    # Both require h==H and w==2*W
    if h == H and w == 2 * W:
        X_flip = np.fliplr(X)

        # Variant: rev|id => [fliplr(X) | X]
        if (np.array_equal(Y[:, :W], X_flip) and
            np.array_equal(Y[:, W:], X)):
            candidates.append(("h-mirror-concat", ("rev|id",)))

        # Variant: id|rev => [X | fliplr(X)]
        if (np.array_equal(Y[:, :W], X) and
            np.array_equal(Y[:, W:], X_flip)):
            candidates.append(("h-mirror-concat", ("id|rev",)))

    # 3. v-double: vertical duplicate [X; X]
    # Requires h==2*H and w==W
    if h == 2 * H and w == W:
        if (np.array_equal(Y[:H, :], X) and
            np.array_equal(Y[H:, :], X)):
            candidates.append(("v-double", None))

    # 4. h-concat-dup: horizontal duplicate [X|X]
    # Requires h==H and w==2*W
    if h == H and w == 2 * W:
        expected = np.concatenate([X, X], axis=1)
        if np.array_equal(Y, expected):
            candidates.append(("h-concat-dup", None))

    # 5. v-concat-dup: vertical duplicate [X;X]
    # Requires h==2*H and w==W
    if h == 2 * H and w == W:
        expected = np.concatenate([X, X], axis=0)
        if np.array_equal(Y, expected):
            candidates.append(("v-concat-dup", None))

    return candidates


def get_detailed_checks(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Compute detailed boolean flags for receipt generation.

    Returns a dict with the structure expected in receipts:
      {
        "identity": {"shape_eq": bool},
        "h_mirror_concat": {"rev|id": {"match": bool}, "id|rev": {"match": bool}},
        "v_double": {"match": bool},
        "h_concat_dup": {"match": bool},
        "v_concat_dup": {"match": bool}
      }

    Args:
        X: Input grid (H, W)
        Y: Output grid (h, w)

    Returns:
        Dict of detailed boolean flags
    """
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)

    H, W = X.shape
    h, w = Y.shape

    # Identity
    identity_shape_eq = (h == H and w == W)

    # h-mirror-concat variants
    h_mirror_concat_rev_id = False
    h_mirror_concat_id_rev = False
    if h == H and w == 2 * W:
        X_flip = np.fliplr(X)
        h_mirror_concat_rev_id = (
            np.array_equal(Y[:, :W], X_flip) and
            np.array_equal(Y[:, W:], X)
        )
        h_mirror_concat_id_rev = (
            np.array_equal(Y[:, :W], X) and
            np.array_equal(Y[:, W:], X_flip)
        )

    # v-double
    v_double_match = False
    if h == 2 * H and w == W:
        v_double_match = (
            np.array_equal(Y[:H, :], X) and
            np.array_equal(Y[H:, :], X)
        )

    # h-concat-dup
    h_concat_dup_match = False
    if h == H and w == 2 * W:
        expected = np.concatenate([X, X], axis=1)
        h_concat_dup_match = np.array_equal(Y, expected)

    # v-concat-dup
    v_concat_dup_match = False
    if h == 2 * H and w == W:
        expected = np.concatenate([X, X], axis=0)
        v_concat_dup_match = np.array_equal(Y, expected)

    return {
        "identity": {"shape_eq": identity_shape_eq},
        "h_mirror_concat": {
            "rev|id": {"match": h_mirror_concat_rev_id},
            "id|rev": {"match": h_mirror_concat_id_rev}
        },
        "v_double": {"match": v_double_match},
        "h_concat_dup": {"match": h_concat_dup_match},
        "v_concat_dup": {"match": v_concat_dup_match}
    }


# ============================================================================
# WO-3B: Types-Periodic Tile Verifier
# ============================================================================


def verify_tile_types(
    X: np.ndarray,
    Y: np.ndarray,
    T_Y: np.ndarray
) -> Optional[Tuple[str, Tuple[int, int]]]:
    """
    Verify types-periodic tiling on type mosaic T_Y.

    Checks if Y is an integer blow-up of X based on types (not colors):
      - Shape constraint: h % H == 0 and w % W == 0
      - Periodicity: T_Y[r,c] == T_Y[r % H, c % W] for all r,c

    This is a FREE morphism verification done on types (Π(Y)), not colors.

    Args:
        X: Input grid (H, W) - used only for shape
        Y: Output grid (h, w) - used only for shape
        T_Y: Type mosaic of Y from Π(Y), shape (h, w)

    Returns:
        ("tile", (sh, sw)) if verified, else None
        where sh = h // H, sw = w // W
    """
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)
    T_Y = np.asarray(T_Y, dtype=np.int32)

    H, W = X.shape
    h, w = Y.shape

    # Step 1: Check if shapes are multiples
    if h % H != 0 or w % W != 0:
        return None

    # Step 2: Compute scale factors
    sh, sw = h // H, w // W

    # Step 3: Extract base type block (first H×W block)
    base = T_Y[0:H, 0:W]

    # Step 4: Build expected tiled types using np.tile
    T_expected = np.tile(base, (sh, sw))

    # Step 5: Check exact equality
    if np.array_equal(T_Y, T_expected):
        return ("tile", (sh, sw))
    else:
        return None


def get_tile_detailed_checks(
    X: np.ndarray,
    Y: np.ndarray,
    T_Y: np.ndarray
) -> dict:
    """
    Compute detailed tile verification info for receipt generation.

    Returns dict with:
      {
        "H": int, "W": int, "h": int, "w": int,
        "sh": int, "sw": int,
        "periodic_by_types": bool,
        "method": str  # "tile"
      }

    Args:
        X: Input grid (H, W)
        Y: Output grid (h, w)
        T_Y: Type mosaic of Y

    Returns:
        Dict of detailed verification info
    """
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)
    T_Y = np.asarray(T_Y, dtype=np.int32)

    H, W = X.shape
    h, w = Y.shape

    # Default values
    sh, sw = None, None
    periodic_by_types = False

    # Check if shapes are multiples
    if h % H == 0 and w % W == 0:
        sh, sw = h // H, w // W

        # Extract base and build expected
        base = T_Y[0:H, 0:W]
        T_expected = np.tile(base, (sh, sw))

        # Check periodicity
        periodic_by_types = np.array_equal(T_Y, T_expected)

    return {
        "H": int(H),
        "W": int(W),
        "h": int(h),
        "w": int(w),
        "sh": int(sh) if sh is not None else None,
        "sw": int(sw) if sw is not None else None,
        "periodic_by_types": periodic_by_types,
        "method": "tile"
    }
