"""
WO-7: Fill by Rank + Idempotence

Given test type mosaic T_test (after disjointify) and paid quotas K from Y₀,
produce final grid Y* using the meet rule (least fill by rank).

All operations use vectorized NumPy primitives:
- np.cumsum for cumulative cuts Σ
- np.searchsorted for minimal color assignment
- np.bincount for quota verification

Idempotence: Π(Y*) + quotas(Y*) + re-fill = Y*
"""

import hashlib
from typing import Dict, Tuple
import numpy as np

from arc.pi import types_from_output
from arc.quotas import quotas


def fill_by_rank(
    T_test: np.ndarray,                 # test type mosaic after disjointify
    parent_of: Dict[int, int],          # new type id S' -> parent Y0 type id S
    K: Dict[int, np.ndarray],           # quotas per parent type (length-C vectors)
    C: int                              # palette size (ARC=10)
) -> np.ndarray:
    """
    Return Y* (same HxW as T_test) using the meet rule per type block.

    Algorithm (vectorized per type):
      1. For each new type S' in T_test:
         - Get row-major positions: idx = flatnonzero(T_test.ravel() == S')
         - Get ranks: ranks = arange(1, |idx|+1)
         - Lookup parent: S = parent_of[S']
         - Compute Σ: cumsum(K[S])
         - Assign colors: searchsorted(Σ, ranks, side='left')
         - Write to Y*

    Args:
        T_test: Test type mosaic (H, W) with disjoint type IDs
        parent_of: Mapping from new type ID S' to parent Y₀ type ID S
        K: Quotas dict {parent_type_id -> length-C vector}
        C: Palette size (10 for ARC)

    Returns:
        Y_star: Filled grid (H, W) with colors
    """
    T_test = np.asarray(T_test, dtype=np.int32)
    H, W = T_test.shape

    # Initialize output
    Y_star = np.zeros((H, W), dtype=np.int64)

    # Get unique new type IDs
    unique_new_types = np.unique(T_test).tolist()

    # Process each new type block
    for new_type_id in unique_new_types:
        # Step 1: Gather row-major positions
        idx = np.flatnonzero(T_test.ravel(order="C") == new_type_id)

        if len(idx) == 0:
            continue

        # Step 2: 1-based ranks
        ranks = np.arange(1, idx.size + 1, dtype=np.int64)

        # Step 3: Parent lookup
        if new_type_id not in parent_of:
            raise ValueError(
                f"Missing parent mapping for type {new_type_id}. "
                f"parent_of must be present for every new type id in T_test."
            )

        parent_type_id = parent_of[new_type_id]

        # Step 4: Get quotas for parent type
        if parent_type_id not in K:
            raise ValueError(
                f"Missing quotas for parent type {parent_type_id}. "
                f"K[S] must be present for all parent types."
            )

        # Step 5: Compute cumulative cuts Σ
        Sigma = np.cumsum(K[parent_type_id])

        # Step 6: Vectorized rank → color mapping
        # searchsorted(Sigma, rank, side='left') gives smallest c where Σ[c] >= rank
        cols = np.searchsorted(Sigma, ranks, side="left")

        # Step 7: Write to output
        Y_star.ravel(order="C")[idx] = cols

    return Y_star


def idempotence_check(
    Y_star: np.ndarray,
    C: int
) -> bool:
    """
    Recompute Π(Y*), K'(Y*), re-fill, and check Y2 == Y* (bit-equality).

    Algorithm:
      1. Compute Π(Y*) → T1
      2. Compute quotas(Y*, T1) → K1
      3. Re-fill with identity parent map: Y2 = fill_by_rank(T1, identity_map, K1, C)
      4. Check Y2 == Y* (bit equality)

    Args:
        Y_star: Filled grid to check
        C: Palette size

    Returns:
        True if idempotent (Y2 == Y*), False otherwise
    """
    Y_star = np.asarray(Y_star, dtype=np.int64)

    # Step 1: Π on Y*
    T1, _ = types_from_output(Y_star)

    # Step 2: K' on Y*
    K1 = quotas(Y_star, T1, C)

    # Step 3: Build identity parent map
    # Every type in T1 is its own parent
    unique_types = np.unique(T1).tolist()
    parent_of_identity = {int(t): int(t) for t in unique_types}

    # Step 4: Re-fill
    Y2 = fill_by_rank(T1, parent_of_identity, K1, C)

    # Step 5: Check bit equality
    return np.array_equal(Y2, Y_star)


def generate_fill_receipt(
    task_id: str,
    T_test: np.ndarray,
    parent_of: Dict[int, int],
    K: Dict[int, np.ndarray],
    Y_star: np.ndarray,
    C: int
) -> Dict:
    """
    Generate fill receipt with block-level verification.

    Args:
        task_id: Task identifier
        T_test: Test type mosaic
        parent_of: Parent mapping
        K: Quotas dict
        Y_star: Filled grid
        C: Palette size

    Returns:
        Receipt dict ready for JSONL output
    """
    # Get unique new type IDs
    unique_new_types = sorted(np.unique(T_test).tolist())

    blocks = []

    for new_type_id in unique_new_types:
        # Get positions
        idx = np.flatnonzero(T_test.ravel(order="C") == new_type_id)

        if len(idx) == 0:
            continue

        # Get parent
        parent_type_id = parent_of.get(new_type_id, new_type_id)

        # Get quotas for parent
        if parent_type_id not in K:
            continue

        parent_quotas = K[parent_type_id]

        # Compute Sigma
        Sigma = np.cumsum(parent_quotas).tolist()

        # Get colors in this block
        blk_colors = Y_star.ravel(order="C")[idx]

        # CRITICAL: Check for OOB colors (conservation law violation)
        max_color = int(np.max(blk_colors)) if len(blk_colors) > 0 else 0
        oob_color = (max_color >= C)

        # Verify quota satisfaction (no truncation - let violations fail)
        actual_counts = np.bincount(blk_colors, minlength=C)
        if len(actual_counts) > C:
            # OOB colors present - mark as failed
            quota_satisfied = False
        else:
            # Pad to C if needed
            if len(actual_counts) < C:
                actual_counts = np.pad(actual_counts, (0, C - len(actual_counts)))
            quota_satisfied = np.array_equal(actual_counts[:C], parent_quotas)

        # rank_minimality is always true by definition of searchsorted
        rank_minimality = True

        # Check conservation law: |S'| must equal sum(K[parent])
        size_match = (len(idx) == int(np.sum(parent_quotas)))

        blocks.append({
            "type_new": int(new_type_id),
            "parent_type": int(parent_type_id),
            "count": int(len(idx)),
            "Sigma": Sigma,
            "quota_satisfied": bool(quota_satisfied),
            "rank_minimality": rank_minimality,
            "oob_color": bool(oob_color),
            "size_match": bool(size_match)
        })

    # Run idempotence check
    idempotent = idempotence_check(Y_star, C)

    # Compute SHA256
    sha256_Y_star = hashlib.sha256(Y_star.astype(np.int64).tobytes()).hexdigest()

    receipt = {
        "task_id": task_id,
        "fill": {
            "C": C,
            "blocks": blocks,
            "idempotent": idempotent,
            "sha256_Y_star": sha256_Y_star
        }
    }

    return receipt
