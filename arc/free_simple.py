"""
WO-3A: FREE Simple Verifiers

Deterministic verification of simple FREE terminals at color level:
  - identity (shape equality)
  - h-mirror-concat (2 variants)
  - v-double (vertical duplicate)
  - h-concat-dup (horizontal duplicate)
  - v-concat-dup (vertical duplicate)

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
