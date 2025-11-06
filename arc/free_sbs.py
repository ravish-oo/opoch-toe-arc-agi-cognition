"""
WO-3C & WO-3D: FREE Verifiers - SBS (Selector-driven Block Substitution)

WO-3C - SBS-Y (templates from Π(Y)):
  - Each block position (i,j) corresponds to input position X[i,j]
  - Input value at X[i,j] acts as a selector choosing which type template to place
  - Templates are actual (sh×sw) blocks extracted from T_Y at first occurrence
  - All positions in X with the same value must have identical type blocks in Y

WO-3D - SBS-Param (templates from Π(X)):
  - Same block grid structure as SBS-Y
  - Templates are CONSTANT (sh×sw) arrays filled with single T_X type value
  - Parametric: input X defines both selector AND template patterns
  - All positions in X with the same value must have identical type blocks in Y

Both verifiers work on TYPES only, not colors.
"""

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import Π from WO-1 for SBS-Param
from arc.pi import types_from_output


def verify_SBS_Y(
    X: np.ndarray,
    T_Y: np.ndarray
) -> Optional[Tuple[str, Tuple[int, int, Dict[int, int], Dict[int, str]]]]:
    """
    Prove SBS-Y on types. Return:
      ("SBS-Y", (sh, sw, sigma_table, template_hashes))
    or None if not proven.

    SBS-Y verification checks:
      1. h = sh * H and w = sw * W (integer block factors)
      2. For each unique value v in X, all blocks at positions (i,j)
         where X[i,j]==v are exactly equal (same type pattern)
      3. This builds a finite selector σ: color → template_id

    Args:
        X: Input grid (H, W) - provides selector values
        T_Y: Type mosaic of Y from Π(Y), shape (h, w)

    Returns:
        ("SBS-Y", (sh, sw, sigma_table, template_hashes)) if proven, else None
        where:
          - sh, sw: block scale factors
          - sigma_table: dict {palette_value -> template_id}
          - template_hashes: dict {template_id -> sha256 hex}
    """
    X = np.asarray(X, dtype=np.int32)
    T_Y = np.asarray(T_Y, dtype=np.int32)

    H, W = X.shape
    h, w = T_Y.shape

    # Step 1: Check divisibility
    if h % H != 0 or w % W != 0:
        return None

    sh, sw = h // H, w // W

    # Step 2: Partition T_Y into blocks (types only)
    # Reshape to (H, sh, W, sw) then moveaxis to (H, W, sh, sw)
    blocks = T_Y.reshape(H, sh, W, sw)
    blocks = np.moveaxis(blocks, 1, 2)  # now blocks[i,j] is (sh, sw)

    # Step 3: Build σ and templates
    # Collect unique values in X (row-major order)
    unique_values = []
    seen = set()
    for i in range(H):
        for j in range(W):
            v = int(X[i, j])
            if v not in seen:
                unique_values.append(v)
                seen.add(v)

    # For each unique value, find first occurrence and verify all match
    templates = {}  # v -> block array

    for v in unique_values:
        # Find first occurrence of v in X (row-major order)
        first_i, first_j = None, None
        for i in range(H):
            for j in range(W):
                if X[i, j] == v:
                    first_i, first_j = i, j
                    break
            if first_i is not None:
                break

        # Set template for this value
        templates[v] = blocks[first_i, first_j].copy()

        # Verify all other occurrences match the template
        for i in range(H):
            for j in range(W):
                if X[i, j] == v:
                    if not np.array_equal(blocks[i, j], templates[v]):
                        # Mismatch found - SBS-Y not proven
                        return None

    # Step 4: Build sigma_table and template_hashes
    # Assign template IDs in order of first occurrence
    template_id_map = {v: idx for idx, v in enumerate(unique_values)}
    sigma_table = {int(v): template_id_map[v] for v in unique_values}

    # Compute hashes for templates (by template_id)
    template_hashes = {}
    for v in unique_values:
        tid = template_id_map[v]
        block_bytes = templates[v].tobytes()
        template_hashes[tid] = hashlib.sha256(block_bytes).hexdigest()

    # Step 5: Return proof object
    return ("SBS-Y", (sh, sw, sigma_table, template_hashes))


def get_sbs_y_detailed_checks(
    X: np.ndarray,
    T_Y: np.ndarray
) -> dict:
    """
    Compute detailed SBS-Y verification info for receipt generation.

    Returns dict with:
      {
        "H": int, "W": int, "h": int, "w": int,
        "sh": int, "sw": int,
        "palette_values_in_X": list[int],
        "sigma_table": dict (if proven) or None,
        "template_hashes": dict (if proven) or None,
        "blocks_match_all": bool,
        "mismatch_example": dict (if mismatch found) or None
      }

    Args:
        X: Input grid (H, W)
        T_Y: Type mosaic of Y

    Returns:
        Dict of detailed verification info
    """
    X = np.asarray(X, dtype=np.int32)
    T_Y = np.asarray(T_Y, dtype=np.int32)

    H, W = X.shape
    h, w = T_Y.shape

    # Collect palette values in X
    palette_values = []
    seen = set()
    for i in range(H):
        for j in range(W):
            v = int(X[i, j])
            if v not in seen:
                palette_values.append(v)
                seen.add(v)

    # Default values
    sh, sw = None, None
    blocks_match_all = False
    sigma_table = None
    template_hashes = None
    mismatch_example = None

    # Check divisibility
    if h % H == 0 and w % W == 0:
        sh, sw = h // H, w // W

        # Partition into blocks
        blocks = T_Y.reshape(H, sh, W, sw)
        blocks = np.moveaxis(blocks, 1, 2)

        # Build templates and verify matches
        templates = {}
        all_match = True

        for v in palette_values:
            # Find first occurrence
            first_i, first_j = None, None
            for i in range(H):
                for j in range(W):
                    if X[i, j] == v:
                        first_i, first_j = i, j
                        break
                if first_i is not None:
                    break

            templates[v] = blocks[first_i, first_j].copy()

            # Check all occurrences
            for i in range(H):
                for j in range(W):
                    if X[i, j] == v:
                        if not np.array_equal(blocks[i, j], templates[v]):
                            all_match = False
                            if mismatch_example is None:
                                mismatch_example = {"v": int(v), "i": int(i), "j": int(j)}
                            break
                if not all_match and mismatch_example is not None:
                    break

            if not all_match:
                break

        blocks_match_all = all_match

        # If all match, build sigma_table and hashes
        if blocks_match_all:
            template_id_map = {v: idx for idx, v in enumerate(palette_values)}
            sigma_table = {int(v): template_id_map[v] for v in palette_values}

            template_hashes = {}
            for v in palette_values:
                tid = template_id_map[v]
                block_bytes = templates[v].tobytes()
                template_hashes[tid] = hashlib.sha256(block_bytes).hexdigest()

    return {
        "H": int(H),
        "W": int(W),
        "h": int(h),
        "w": int(w),
        "sh": int(sh) if sh is not None else None,
        "sw": int(sw) if sw is not None else None,
        "palette_values_in_X": palette_values,
        "sigma_table": sigma_table,
        "template_hashes": template_hashes,
        "blocks_match_all": blocks_match_all,
        "mismatch_example": mismatch_example
    }


# ============================================================================
# WO-3D: SBS-Param (templates from Π(X))
# ============================================================================


def verify_SBS_param(
    X: np.ndarray,
    Y: np.ndarray
) -> Optional[Tuple[str, Tuple[int, int, Dict[int, int], Dict[int, str]]]]:
    """
    Prove SBS-Param on types (templates from Π(X)). Return:
      ("SBS-param", (sh, sw, sigma_table, template_hashes))
    or None if not proven.

    SBS-Param verification checks:
      1. Compute T_X = Π(X) and T_Y = Π(Y)
      2. h = sh * H and w = sw * W (integer block factors)
      3. For each unique value v in X, build CONSTANT template from Π(X):
         template_v = np.full((sh, sw), T_X[first_i, first_j])
      4. All blocks at positions (i,j) where X[i,j]==v must equal template_v

    This is the parametric variant: input X defines both selector AND templates.

    Args:
        X: Input grid (H, W) - provides selector values AND template source
        Y: Output grid (h, w)

    Returns:
        ("SBS-param", (sh, sw, sigma_table, template_hashes)) if proven, else None
        where:
          - sh, sw: block scale factors
          - sigma_table: dict {palette_value -> template_id}
          - template_hashes: dict {template_id -> sha256 hex}
    """
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)

    # Compute type mosaics from Π
    T_X, _ = types_from_output(X)
    T_Y, _ = types_from_output(Y)

    H, W = X.shape
    h, w = Y.shape

    # Step 1: Check divisibility
    if h % H != 0 or w % W != 0:
        return None

    sh, sw = h // H, w // W

    # Step 2: Partition T_Y into blocks (types only)
    # Reshape to (H, sh, W, sw) then moveaxis to (H, W, sh, sw)
    blocks = T_Y.reshape(H, sh, W, sw)
    blocks = np.moveaxis(blocks, 1, 2)  # now blocks[i,j] is (sh, sw)

    # Step 3: Build templates from Π(X) - CANONICAL
    # Collect unique values in X (row-major order)
    unique_values = []
    seen = set()
    for i in range(H):
        for j in range(W):
            v = int(X[i, j])
            if v not in seen:
                unique_values.append(v)
                seen.add(v)

    # For each unique value, build constant template from T_X
    templates = {}  # v -> template array (sh, sw)

    for v in unique_values:
        # Find first occurrence of v in X (row-major order)
        first_i, first_j = None, None
        for i in range(H):
            for j in range(W):
                if X[i, j] == v:
                    first_i, first_j = i, j
                    break
            if first_i is not None:
                break

        # Build CONSTANT template from Π(X) at first occurrence
        # This is the canonical definition: replicate the T_X type value
        templates[v] = np.full((sh, sw), T_X[first_i, first_j], dtype=T_X.dtype)

        # Verify all other occurrences match the template
        for i in range(H):
            for j in range(W):
                if X[i, j] == v:
                    if not np.array_equal(blocks[i, j], templates[v]):
                        # Mismatch found - SBS-Param not proven
                        return None

    # Step 4: Build sigma_table and template_hashes
    # Assign template IDs in order of first occurrence
    template_id_map = {v: idx for idx, v in enumerate(unique_values)}
    sigma_table = {int(v): template_id_map[v] for v in unique_values}

    # Compute hashes for templates (by template_id)
    template_hashes = {}
    for v in unique_values:
        tid = template_id_map[v]
        block_bytes = templates[v].tobytes()
        template_hashes[tid] = hashlib.sha256(block_bytes).hexdigest()

    # Step 5: Return proof object
    return ("SBS-param", (sh, sw, sigma_table, template_hashes))


def get_sbs_param_detailed_checks(
    X: np.ndarray,
    Y: np.ndarray
) -> dict:
    """
    Compute detailed SBS-Param verification info for receipt generation.

    Returns dict with:
      {
        "H": int, "W": int, "h": int, "w": int,
        "sh": int, "sw": int,
        "palette_values_in_X": list[int],
        "sigma_table": dict (if proven) or None,
        "template_hashes": dict (if proven) or None,
        "blocks_match_all": bool,
        "mismatch_example": dict (if mismatch found) or None
      }

    Args:
        X: Input grid (H, W)
        Y: Output grid (h, w)

    Returns:
        Dict of detailed verification info
    """
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)

    # Compute type mosaics
    T_X, _ = types_from_output(X)
    T_Y, _ = types_from_output(Y)

    H, W = X.shape
    h, w = Y.shape

    # Collect palette values in X
    palette_values = []
    seen = set()
    for i in range(H):
        for j in range(W):
            v = int(X[i, j])
            if v not in seen:
                palette_values.append(v)
                seen.add(v)

    # Default values
    sh, sw = None, None
    blocks_match_all = False
    sigma_table = None
    template_hashes = None
    mismatch_example = None

    # Check divisibility
    if h % H == 0 and w % W == 0:
        sh, sw = h // H, w // W

        # Partition into blocks
        blocks = T_Y.reshape(H, sh, W, sw)
        blocks = np.moveaxis(blocks, 1, 2)

        # Build templates from Π(X) and verify matches
        templates = {}
        all_match = True

        for v in palette_values:
            # Find first occurrence
            first_i, first_j = None, None
            for i in range(H):
                for j in range(W):
                    if X[i, j] == v:
                        first_i, first_j = i, j
                        break
                if first_i is not None:
                    break

            # Build constant template from Π(X)
            templates[v] = np.full((sh, sw), T_X[first_i, first_j], dtype=T_X.dtype)

            # Check all occurrences
            for i in range(H):
                for j in range(W):
                    if X[i, j] == v:
                        if not np.array_equal(blocks[i, j], templates[v]):
                            all_match = False
                            if mismatch_example is None:
                                mismatch_example = {"v": int(v), "i": int(i), "j": int(j)}
                            break
                if not all_match and mismatch_example is not None:
                    break

            if not all_match:
                break

        blocks_match_all = all_match

        # If all match, build sigma_table and hashes
        if blocks_match_all:
            template_id_map = {v: idx for idx, v in enumerate(palette_values)}
            sigma_table = {int(v): template_id_map[v] for v in palette_values}

            template_hashes = {}
            for v in palette_values:
                tid = template_id_map[v]
                block_bytes = templates[v].tobytes()
                template_hashes[tid] = hashlib.sha256(block_bytes).hexdigest()

    return {
        "H": int(H),
        "W": int(W),
        "h": int(h),
        "w": int(w),
        "sh": int(sh) if sh is not None else None,
        "sw": int(sw) if sw is not None else None,
        "palette_values_in_X": palette_values,
        "sigma_table": sigma_table,
        "template_hashes": template_hashes,
        "blocks_match_all": blocks_match_all,
        "mismatch_example": mismatch_example
    }
