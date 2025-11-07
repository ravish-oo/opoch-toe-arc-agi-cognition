"""
WO-5: Transport Types + Disjointify

Given a proven FREE terminal from WO-4, transport training output types (T_Y0)
to create test output types (T_test), then disjointify to ensure fills cannot
bleed across replicated blocks.

All operations work on TYPES (from Π), never colors.

Transport terminals:
  - identity: T_test = T_Y0
  - h-mirror-concat: horizontal mirror concatenation
  - v-double: vertical duplication
  - h/v-concat-dup: horizontal/vertical duplication
  - tile: integer blow-up
  - SBS-Y: selector-driven block substitution from Π(Y) templates
  - SBS-param: selector-driven block substitution from Π(X) templates

Disjointify: 4-connected component labeling per type ID to prevent fills
from bleeding across replicated blocks.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from skimage.measure import label as skimage_label


def transport_types(
    T_train: np.ndarray,
    free_tuple: Tuple[str, Tuple[Any, ...]],
    X_test_shape: Tuple[int, int],
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Build T_test per the proven FREE terminal (types-only), then disjointify.

    Args:
        T_train: Training output type mosaic Π(Y0), shape (h0, w0)
        free_tuple: (kind, params) from WO-4, e.g., ("tile", (sh, sw))
        X_test_shape: Test input shape (H, W)
        X_test: Test input (needed for SBS-* selection only)

    Returns:
        T_test_disjoint: Test output type mosaic after disjointify (np.ndarray[int])
        parent_of: Dict[int, int] mapping new type id S' -> parent Y0 type id S
    """
    kind, params = free_tuple

    # Apply FREE morphism (types only, no colors)
    # Replicating transports return (T_test, copy_mask), identity returns T_test
    if kind == "identity":
        T_test = _transport_identity(T_train)
        copy_mask = None
    elif kind == "h-mirror-concat":
        T_test, copy_mask = _transport_h_mirror_concat(T_train, params)
    elif kind == "v-double":
        T_test, copy_mask = _transport_v_double(T_train)
    elif kind == "h-concat-dup":
        T_test, copy_mask = _transport_h_concat_dup(T_train)
    elif kind == "v-concat-dup":
        T_test, copy_mask = _transport_v_concat_dup(T_train)
    elif kind == "tile":
        T_test, copy_mask = _transport_tile(T_train, params)
    elif kind == "SBS-Y":
        T_test, copy_mask = _transport_sbs(T_train, params, X_test, kind="SBS-Y")
    elif kind == "SBS-param":
        T_test, copy_mask = _transport_sbs(T_train, params, X_test, kind="SBS-param")
    else:
        raise ValueError(f"Unknown FREE kind: {kind}")

    # Disjointify if terminal involves replication
    if _needs_disjointify(kind):
        # Pass copy_mask to prevent cross-copy merging
        T_test_disjoint, parent_of = disjointify(T_test, copy_mask)
    else:
        # Identity doesn't replicate - use identity parent mapping
        T_test_disjoint = T_test
        parent_of = {int(tid): int(tid) for tid in np.unique(T_test)}

    return T_test_disjoint, parent_of


def _needs_disjointify(kind: str) -> bool:
    """Check if terminal needs disjointification."""
    # Identity doesn't replicate, so no disjointify needed
    return kind != "identity"


def _transport_identity(T_train: np.ndarray) -> np.ndarray:
    """Identity: T_test = T_train (copy)."""
    return T_train.copy()


def _transport_h_mirror_concat(T_train: np.ndarray, params: Tuple[Any, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horizontal mirror concatenation.

    Two variants:
      - "rev|id": [fliplr(T_train) | T_train]
      - "id|rev": [T_train | fliplr(T_train)]

    Returns:
        T_test: Concatenated type mosaic
        copy_mask: Grid marking left half as copy 0, right half as copy 1
    """
    variant = params[0] if params else "id|rev"
    T_flip = np.fliplr(T_train)
    h, w = T_train.shape

    if variant == "rev|id":
        T_test = np.concatenate([T_flip, T_train], axis=1)
    else:  # "id|rev"
        T_test = np.concatenate([T_train, T_flip], axis=1)

    # Create copy mask: left half = 0, right half = 1
    copy_mask = np.zeros((h, 2 * w), dtype=np.int32)
    copy_mask[:, w:] = 1

    return T_test, copy_mask


def _transport_v_double(T_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vertical double: [T_train; T_train].

    Returns:
        T_test: Vertically duplicated type mosaic
        copy_mask: Grid marking top half as copy 0, bottom half as copy 1
    """
    T_test = np.concatenate([T_train, T_train], axis=0)
    h, w = T_train.shape

    # Create copy mask: top half = 0, bottom half = 1
    copy_mask = np.zeros((2 * h, w), dtype=np.int32)
    copy_mask[h:, :] = 1

    return T_test, copy_mask


def _transport_h_concat_dup(T_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horizontal concatenation duplicate: [T_train | T_train].

    Returns:
        T_test: Horizontally duplicated type mosaic
        copy_mask: Grid marking left half as copy 0, right half as copy 1
    """
    T_test = np.concatenate([T_train, T_train], axis=1)
    h, w = T_train.shape

    # Create copy mask: left half = 0, right half = 1
    copy_mask = np.zeros((h, 2 * w), dtype=np.int32)
    copy_mask[:, w:] = 1

    return T_test, copy_mask


def _transport_v_concat_dup(T_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vertical concatenation duplicate: [T_train; T_train].

    Returns:
        T_test: Vertically duplicated type mosaic
        copy_mask: Grid marking top half as copy 0, bottom half as copy 1
    """
    T_test = np.concatenate([T_train, T_train], axis=0)
    h, w = T_train.shape

    # Create copy mask: top half = 0, bottom half = 1
    copy_mask = np.zeros((2 * h, w), dtype=np.int32)
    copy_mask[h:, :] = 1

    return T_test, copy_mask


def _transport_tile(T_train: np.ndarray, params: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tile (integer blow-up): repeat T_train by (sh, sw).

    Args:
        T_train: Base type pattern
        params: (sh, sw) - repetition factors

    Returns:
        T_test: Tiled type mosaic of shape (sh*h, sw*w)
        copy_mask: Grid where each tile block has a unique copy_id
    """
    sh, sw = params
    h, w = T_train.shape

    T_test = np.tile(T_train, (sh, sw))

    # Create copy mask: each tile block (i,j) gets unique copy_id
    copy_mask = np.zeros((sh * h, sw * w), dtype=np.int32)
    copy_id = 0
    for i in range(sh):
        for j in range(sw):
            r0, r1 = i * h, (i + 1) * h
            c0, c1 = j * w, (j + 1) * w
            copy_mask[r0:r1, c0:c1] = copy_id
            copy_id += 1

    return T_test, copy_mask


def _transport_sbs(
    T_train: np.ndarray,
    params: Tuple[Any, ...],
    X_test: np.ndarray,
    kind: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SBS (Selector-driven Block Substitution).

    For SBS-Y: templates from Π(Y)
    For SBS-param: templates from Π(X)

    Args:
        T_train: Training output types (used for templates in SBS-Y)
        params: (sh, sw, sigma_table, template_dict) where template_dict maps tid -> (sh, sw) array
        X_test: Test input for selector
        kind: "SBS-Y" or "SBS-param"

    Returns:
        T_test: Type mosaic with templates placed according to X_test selector
        copy_mask: Grid where each block (i,j) gets unique copy_id
    """
    if len(params) < 3:
        raise ValueError(f"SBS params must include (sh, sw, sigma_table, templates): got {params}")

    sh, sw = params[0], params[1]
    sigma_table = params[2]
    templates = params[3] if len(params) > 3 else {}

    H, W = X_test.shape
    h, w = H * sh, W * sw

    # Build empty test canvas and copy mask
    T_test = np.zeros((h, w), dtype=np.int32)
    copy_mask = np.zeros((h, w), dtype=np.int32)

    # Place templates according to X_test selector
    copy_id = 0
    for i in range(H):
        for j in range(W):
            v = int(X_test[i, j])

            # Assign unique copy_id to this block
            r0, r1 = i * sh, (i + 1) * sh
            c0, c1 = j * sw, (j + 1) * sw
            copy_mask[r0:r1, c0:c1] = copy_id
            copy_id += 1

            # Lookup template ID from sigma
            if v not in sigma_table:
                # Value not in sigma table - use default (zeros)
                continue

            tid = sigma_table[v]

            if tid not in templates:
                # Template not provided - use zeros
                continue

            # Place template into block
            T_test[r0:r1, c0:c1] = templates[tid]

    return T_test, copy_mask


def disjointify(T: np.ndarray, copy_mask: np.ndarray = None) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Disjointify: 4-connected component labeling per (type ID, copy ID) pair.

    After replication (tile, dup, mirror, SBS), split each type ID into
    separate components to prevent fills from bleeding across block boundaries.

    CRITICAL: When copy_mask is provided, never merge components across different
    copy_ids - this enforces the conservation law (|S'| = sum K[parent]).

    Args:
        T: Type mosaic with replicated blocks (pre-disjoint)
        copy_mask: Optional grid same shape as T, where copy_mask[r,c] = copy_id.
                   Components will never merge across different copy_ids.

    Returns:
        T_disjoint: Type mosaic with 4-connected components labeled uniquely
        parent_of: Dict mapping new type id S' -> parent (pre-disjoint) type id S
    """
    T = np.asarray(T, dtype=np.int32)
    h, w = T.shape

    if copy_mask is None:
        # No copy boundaries - original behavior
        copy_mask = np.zeros_like(T)

    copy_mask = np.asarray(copy_mask, dtype=np.int32)

    # Find unique (type_id, copy_id) pairs in sorted order (for determinism)
    unique_pairs = []
    for r in range(h):
        for c in range(w):
            pair = (int(T[r, c]), int(copy_mask[r, c]))
            if pair not in unique_pairs:
                unique_pairs.append(pair)
    unique_pairs.sort()

    # Build new type mosaic with disjoint labels and parent mapping
    T_disjoint = np.zeros_like(T)
    parent_of = {}
    next_global_id = 0

    # Process each (parent_type_id, copy_id) pair in sorted order
    for parent_type_id, copy_id in unique_pairs:
        # Isolate this (type, copy) combination
        mask = ((T == parent_type_id) & (copy_mask == copy_id)).astype(np.uint8)

        # 4-connected component labeling within this copy
        labels = skimage_label(mask, connectivity=1)

        # Get unique component labels (exclude 0 = background)
        component_ids = np.unique(labels)
        component_ids = component_ids[component_ids > 0]

        # Assign new global IDs to each component
        for comp_id in component_ids:
            comp_mask = (labels == comp_id)
            T_disjoint[comp_mask] = next_global_id
            parent_of[next_global_id] = int(parent_type_id)
            next_global_id += 1

    return T_disjoint, parent_of


def verify_blocks_match(
    T_test: np.ndarray,
    templates: Dict[int, np.ndarray],
    X_test: np.ndarray,
    sigma_table: Dict[int, int],
    sh: int,
    sw: int
) -> bool:
    """
    Verify that every block in T_test matches its template.

    Used for receipt generation to prove correctness before disjointify.

    Args:
        T_test: Test type mosaic before disjointify
        templates: Template dict {tid -> (sh, sw) array}
        X_test: Test input for selector
        sigma_table: Selector mapping {value -> tid}
        sh, sw: Block dimensions

    Returns:
        True if all blocks match their templates exactly
    """
    H, W = X_test.shape

    for i in range(H):
        for j in range(W):
            v = int(X_test[i, j])

            if v not in sigma_table:
                continue

            tid = sigma_table[v]

            if tid not in templates:
                continue

            # Extract block
            r0, r1 = i * sh, (i + 1) * sh
            c0, c1 = j * sw, (j + 1) * sw
            block = T_test[r0:r1, c0:c1]

            # Check exact equality
            if not np.array_equal(block, templates[tid]):
                return False

    return True
