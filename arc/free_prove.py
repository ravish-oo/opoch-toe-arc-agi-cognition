"""
WO-4: FREE Intersection + Pick (Frozen Order)

For each task with multiple training pairs, this module:
  1. Collects all FREE candidates from WO-3A..3D per pair
  2. Intersects candidates across all pairs (strict set intersection)
  3. Applies frozen simplicity order to pick exactly one terminal
  4. Returns either FREE_PROVEN (with chosen candidate) or FREE_UNPROVEN

Frozen simplicity order (rank):
  0: identity
  1: h-mirror-concat, v-double, h-concat-dup, v-concat-dup
  2: band_map
  3: tile
  4: SBS-Y
  5: SBS-param

This is a task-level operation (not per-pair).
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

# Type aliases per WO-4 spec
Kind = str
Params = Optional[Tuple[Any, ...]]
Cand = Tuple[Kind, Params]


def make_hashable(obj: Any) -> Any:
    """
    Convert nested structure to hashable form.

    Dicts and lists become tuples. This allows candidates with dict params
    (e.g., SBS sigma_table) to be added to sets for intersection.

    Args:
        obj: Any object (dict, list, tuple, primitive)

    Returns:
        Hashable version of the object
    """
    if isinstance(obj, dict):
        # Sort by key for determinism
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    elif obj is None:
        return None
    else:
        # Primitives (int, str, etc.) are already hashable
        return obj


def get_order_rank(kind: str) -> int:
    """
    Return frozen simplicity order rank for a FREE kind.

    Frozen order (from WO-4):
      0: identity
      1: h-mirror-concat, v-double, h-concat-dup, v-concat-dup
      2: band_map
      3: tile
      4: SBS-Y
      5: SBS-param

    Args:
        kind: FREE morphism kind string

    Returns:
        Integer rank (0-5, or 99 for unknown)
    """
    order = {
        "identity": 0,
        "h-mirror-concat": 1,
        "v-double": 1,
        "h-concat-dup": 1,
        "v-concat-dup": 1,
        "band_map": 2,
        "tile": 3,
        "SBS-Y": 4,
        "SBS-param": 5,
    }
    return order.get(kind, 99)


def prove_free(
    task: Dict[str, Any],
    per_pair_simple: List[List[Cand]],
    per_pair_tile: List[Optional[Cand]],
    per_pair_sbs_y: List[Optional[Cand]],
    per_pair_sbs_p: List[Optional[Cand]],
    per_pair_stripe: List[List[Cand]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Prove FREE morphism for a task by intersecting per-pair candidates.

    Algorithm:
      1. For each pair, collect all candidates (simple + stripe + tile + SBS-Y + SBS-param)
      2. Normalize to hashable form (dicts → tuples)
      3. Build set per pair
      4. Intersect all sets: candidates must appear in EVERY pair
      5. If empty → FREE_UNPROVEN
      6. Else sort by frozen order, pick first → FREE_PROVEN

    Args:
        task: Task data dict (unused in v0, for future expansion)
        per_pair_simple: List of candidate lists from WO-3A per pair
        per_pair_tile: List of optional tile candidates from WO-3B per pair
        per_pair_sbs_y: List of optional SBS-Y candidates from WO-3C per pair
        per_pair_sbs_p: List of optional SBS-param candidates from WO-3D per pair
        per_pair_stripe: List of candidate lists from WO-3H per pair (optional)

    Returns:
        Tuple of:
          - Status: "FREE_PROVEN" or "FREE_UNPROVEN"
          - Proof data dict with keys:
            - "chosen": Cand (if proven)
            - "intersected": List[Cand]
            - "order_rank": int (if proven)
            - "reason": str (if unproven)
    """
    num_pairs = len(per_pair_simple)

    # Handle optional stripe candidates (default to empty lists)
    if per_pair_stripe is None:
        per_pair_stripe = [[] for _ in range(num_pairs)]

    # Step 1: Build sets per pair
    sets = []
    for pair_idx in range(num_pairs):
        # Collect all candidates for this pair
        candidates = []

        # Simple candidates (WO-3A)
        candidates.extend(per_pair_simple[pair_idx])

        # Stripe candidates (WO-3H)
        candidates.extend(per_pair_stripe[pair_idx])

        # Tile candidate (WO-3B)
        if per_pair_tile[pair_idx] is not None:
            candidates.append(per_pair_tile[pair_idx])

        # SBS-Y candidate (WO-3C)
        if per_pair_sbs_y[pair_idx] is not None:
            candidates.append(per_pair_sbs_y[pair_idx])

        # SBS-param candidate (WO-3D)
        if per_pair_sbs_p[pair_idx] is not None:
            candidates.append(per_pair_sbs_p[pair_idx])

        # Normalize to hashable form
        # Candidates have form (kind, params) where params may contain dicts
        hashable_cands = []
        for kind, params in candidates:
            hashable_params = make_hashable(params)
            hashable_cands.append((kind, hashable_params))

        sets.append(set(hashable_cands))

    # Step 2: Intersect all sets
    if len(sets) == 0:
        # No pairs (shouldn't happen, but handle gracefully)
        intersected_set = set()
    else:
        # Start with first set, intersect with rest
        intersected_set = sets[0]
        for s in sets[1:]:
            intersected_set = intersected_set & s

    # Step 3: Check if intersection is empty
    if len(intersected_set) == 0:
        return ("FREE_UNPROVEN", {
            "reason": "empty_intersection",
            "intersected": []
        })

    # Step 4: Sort by frozen order and pick first
    # Sort key: (order_rank, str(params)) for deterministic tie-breaking
    sorted_cands = sorted(
        list(intersected_set),
        key=lambda cand: (get_order_rank(cand[0]), str(cand[1]))
    )

    chosen = sorted_cands[0]
    chosen_kind, chosen_params = chosen

    # Convert back to list for JSON serialization
    intersected_list = sorted_cands  # Already sorted

    return ("FREE_PROVEN", {
        "chosen": chosen,
        "intersected": intersected_list,
        "order_rank": get_order_rank(chosen_kind)
    })


def compute_chosen_sha256(chosen: Cand) -> str:
    """
    Compute deterministic SHA256 hash of chosen candidate.

    Used for receipt integrity checks.

    Args:
        chosen: (kind, params) tuple

    Returns:
        Hex SHA256 digest string
    """
    # Serialize to JSON with sorted keys for determinism
    chosen_json = json.dumps(chosen, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(chosen_json.encode('utf-8')).hexdigest()
