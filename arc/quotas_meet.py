"""
WO-6b: Multi-Y Quotas Meet (strictly types-aligned, semantic meet)

Compute global meet quotas table (K*) over φ-type classes across all training outputs.
Align types by φ signature (equal φ ⇒ same type class) and take component-wise minimum.

Critical: Quotas come ONLY from training outputs, NEVER from inputs.
"""

import hashlib
import numpy as np
from typing import Dict, Tuple, Any, List

from arc.pi import types_from_output

PhiKey = Tuple[int, ...]  # the φ feature tuple from WO-1 (13-element tuple)


def build_phi_key_index(Y: np.ndarray, T: np.ndarray) -> Dict[int, PhiKey]:
    """
    Map each parent type-id S in Π(Y) to its canonical φ key (equal φ ⇒ same type class).

    Args:
        Y: Output grid (H, W)
        T: Type mosaic Π(Y) (H, W)

    Returns:
        Dict mapping type_id -> φ tuple (13-element)
    """
    Y = np.asarray(Y, dtype=np.int32)
    T = np.asarray(T, dtype=np.int32)

    # Get codebook from types_from_output (φ -> type_id)
    _, codebook = types_from_output(Y)

    # Invert: type_id -> φ
    type_to_phi = {type_id: phi for phi, type_id in codebook.items()}

    return type_to_phi


def quotas_per_phi_on_grid(Y: np.ndarray, T: np.ndarray, C: int) -> Dict[PhiKey, np.ndarray]:
    """
    For one training output Y with its types T, return per-φ color histograms (length-C vectors).

    Args:
        Y: Output grid (H, W)
        T: Type mosaic Π(Y) (H, W)
        C: Number of colors (10 for ARC)

    Returns:
        Dict mapping φ -> length-C vector of color counts
    """
    Y = np.asarray(Y, dtype=np.int32)
    T = np.asarray(T, dtype=np.int32)

    # Get type_id -> φ mapping
    type_to_phi = build_phi_key_index(Y, T)

    # Get unique type IDs
    unique_types = np.unique(T).tolist()

    # Compute per-type quotas, then group by φ
    phi_quotas = {}
    for type_id in unique_types:
        # Get colors for this type
        mask = (T == type_id)
        colors = Y[mask]

        # Count colors with bincount
        counts = np.bincount(colors, minlength=C)[:C].astype(np.int64)

        # Get φ for this type
        phi = type_to_phi[type_id]

        # If multiple types share same φ (shouldn't happen but handle it), sum
        if phi in phi_quotas:
            phi_quotas[phi] = phi_quotas[phi] + counts
        else:
            phi_quotas[phi] = counts

    return phi_quotas


def quotas_meet_all(train_pairs: List[dict], C: int) -> Dict[PhiKey, np.ndarray]:
    """
    For all training outputs in a task, align by φ and compute the component-wise minimum
    ONLY over trainings that contain φ:
      K*(φ, :) = min_{i ∈ I(φ)}  K_i(φ, :)
    where I(φ) = {i : φ appears in Y_i}.

    Critical (A0: no minted differences): Never include zero vectors for absent φ.

    Args:
        train_pairs: List of training pairs (each with "input" and "output" keys)
        C: Number of colors (10 for ARC)

    Returns:
        Dict mapping φ -> K*(φ,:) (length-C vector, element-wise minimum over present trainings)
    """
    # Collect per-training φ quotas
    all_phi_quotas = []  # List of dicts: φ -> vector

    for pair in train_pairs:
        Y_i = np.array(pair["output"], dtype=np.int32)
        T_i, _ = types_from_output(Y_i)

        phi_quotas_i = quotas_per_phi_on_grid(Y_i, T_i, C)
        all_phi_quotas.append(phi_quotas_i)

    # Get union of all φ keys across trainings
    all_phi_keys = set()
    for phi_quotas in all_phi_quotas:
        all_phi_keys.update(phi_quotas.keys())

    # Compute meet for each φ ONLY over trainings where φ appears (A0: no minted differences)
    K_star_phi = {}

    for phi in all_phi_keys:
        # Collect vectors ONLY from trainings that contain this φ
        # I(φ) = {i : φ appears in Y_i}
        vectors = []
        for phi_quotas in all_phi_quotas:
            if phi in phi_quotas:  # CRITICAL: only include present trainings
                vectors.append(phi_quotas[phi])

        # Stack and take element-wise minimum over PRESENT trainings only
        if len(vectors) > 0:
            stack = np.stack(vectors, axis=0)  # shape: (|I(φ)|, C)
            K_star_phi[phi] = np.minimum.reduce(stack, axis=0)

    return K_star_phi


def quotas_for_test_parent(
    parent_of: Dict[int, int],                 # S' -> S, from WO-5
    parent_phi_map: Dict[int, PhiKey],         # S -> φ, from build_phi_key_index on Y0
    K_star_phi: Dict[PhiKey, np.ndarray],      # K*(φ,:) meet quotas
    parent_sizes: Dict[int, int],              # parent_id -> |S| (block size in Y0)
    Y0_quotas: Dict[int, np.ndarray],          # parent_id -> K_Y0(S,:) (fallback quotas)
) -> Tuple[Dict[int, np.ndarray], Dict[PhiKey, Dict[str, Any]]]:
    """
    Produce the final quotas dict for fill: map each child type S' to K*(φ(parent(S')), :)
    with admissibility check (A1: exact balance).

    For each φ, check: sum(K*(φ,c)) == |S_parent(φ)|
    - If true: use K*(φ,:), mark meet_applicable=true
    - If false: fall back to Y0_quotas[parent], mark meet_applicable=false with reason

    Args:
        parent_of: Child type ID -> parent type ID mapping (from WO-5)
        parent_phi_map: Parent type ID -> φ mapping (from build_phi_key_index on Y0)
        K_star_phi: φ -> K*(φ,:) meet quotas (from quotas_meet_all)
        parent_sizes: Parent type ID -> block size (for admissibility check)
        Y0_quotas: Parent type ID -> Y0 quotas (fallback when meet inadmissible)

    Returns:
        Tuple of:
        - K_final: Dict mapping child_type_id -> length-C vector (final quotas for WO-7)
        - phi_metadata: Dict mapping φ -> {meet_applicable, fallback_reason, used_vector, parent_size, sum_meet}
    """
    K_final = {}
    phi_metadata: Dict[PhiKey, Dict[str, Any]] = {}
    C = 10  # ARC palette size

    # First pass: check admissibility per φ
    for parent_id, phi in parent_phi_map.items():
        if phi in phi_metadata:
            continue  # Already checked this φ

        expected_size = parent_sizes.get(parent_id, 0)
        meet_vec = K_star_phi.get(phi, None)

        if meet_vec is None:
            # φ never appeared in any training (shouldn't happen with current logic)
            phi_metadata[phi] = {
                "meet_applicable": False,
                "fallback_reason": "phi_absent",
                "used_vector": "y0_fallback",
                "parent_size": expected_size,
                "sum_meet": 0
            }
        else:
            sum_meet = int(meet_vec.sum())
            if sum_meet == expected_size:
                # Admissible: meet preserves conservation law
                phi_metadata[phi] = {
                    "meet_applicable": True,
                    "fallback_reason": None,
                    "used_vector": "meet",
                    "parent_size": expected_size,
                    "sum_meet": sum_meet
                }
            else:
                # Inadmissible: fall back to Y0 (A1 violation would occur)
                phi_metadata[phi] = {
                    "meet_applicable": False,
                    "fallback_reason": "sum_mismatch",
                    "used_vector": "y0_fallback",
                    "parent_size": expected_size,
                    "sum_meet": sum_meet
                }

    # Second pass: assign quotas to children based on admissibility
    for child_id, parent_id in parent_of.items():
        phi = parent_phi_map.get(parent_id, None)

        if phi is None or phi not in phi_metadata:
            # No φ mapping (shouldn't happen, use Y0 as safe fallback)
            K_final[child_id] = Y0_quotas.get(parent_id, np.zeros(C, dtype=np.int64)).copy()
        else:
            meta = phi_metadata[phi]
            if meta["meet_applicable"]:
                # Use meet quotas
                K_final[child_id] = K_star_phi[phi].copy()
            else:
                # Fall back to Y0 quotas
                K_final[child_id] = Y0_quotas.get(parent_id, np.zeros(C, dtype=np.int64)).copy()

    return K_final, phi_metadata


def generate_quotas_meet_receipt(
    task_id: str,
    task: Dict[str, Any],
    K_star_phi: Dict[PhiKey, np.ndarray],
    parent_phi_map: Dict[int, PhiKey],
    K_final: Dict[int, np.ndarray],
    T_test: np.ndarray,
    parent_of: Dict[int, int],
    phi_metadata: Dict[PhiKey, Dict[str, Any]],
    C: int
) -> Dict[str, Any]:
    """
    Generate quotas-meet receipt for a task (WO-6b) with admissibility diagnostics.

    Args:
        task_id: Task identifier
        task: Task dict with train pairs
        K_star_phi: Meet quotas (φ -> K*(φ,:))
        parent_phi_map: Parent type ID -> φ mapping
        K_final: Final adapted quotas (child_id -> quotas)
        T_test: Test type mosaic
        parent_of: Child -> parent mapping
        phi_metadata: Per-φ metadata with meet_applicable, fallback_reason, etc.
        C: Palette size

    Returns:
        Receipt dict with quotas_meet structure including admissibility diagnostics
    """
    num_trainings = len(task["train"])
    phi_keys_total = len(K_star_phi)
    phi_from_y0 = len(parent_phi_map)

    # Collect samples (first 5 φ keys for verification) with enhanced diagnostics
    samples = []
    for i, (phi, meet_vec) in enumerate(list(K_star_phi.items())[:5]):
        # Hash the φ tuple for compact representation
        phi_hash = hashlib.sha256(str(phi).encode()).hexdigest()[:16]

        # Collect per-training vectors ONLY from trainings containing φ
        train_vectors = []
        present_count = 0
        for pair in task["train"]:
            Y_i = np.array(pair["output"], dtype=np.int32)
            T_i, _ = types_from_output(Y_i)
            phi_quotas_i = quotas_per_phi_on_grid(Y_i, T_i, C)
            if phi in phi_quotas_i:
                train_vectors.append(phi_quotas_i[phi].tolist())
                present_count += 1

        # Get metadata for this φ
        meta = phi_metadata.get(phi, {})

        sample = {
            "phi_hash": phi_hash,
            "present_in": present_count,
            "train_vectors": train_vectors,
            "meet_vector": meet_vec.tolist(),
            "sum_meet": int(meet_vec.sum()),
            "parent_size": meta.get("parent_size", 0),
            "meet_applicable": meta.get("meet_applicable", False),
            "used_vector": meta.get("used_vector", "unknown"),
            "fallback_reason": meta.get("fallback_reason", None)
        }
        samples.append(sample)

    # Verify size_match_pass for test adaptation (should always pass now with fallback)
    size_match_pass = True
    for child_id in np.unique(T_test).tolist():
        child_size = int(np.sum(T_test == child_id))
        quota_sum = int(K_final[child_id].sum())
        if child_size != quota_sum:
            size_match_pass = False
            break

    # Count how many φ used meet vs fallback
    meet_count = sum(1 for meta in phi_metadata.values() if meta["meet_applicable"])
    fallback_count = len(phi_metadata) - meet_count

    receipt = {
        "task_id": task_id,
        "quotas_meet": {
            "C": C,
            "num_trainings": num_trainings,
            "phi_keys_total": phi_keys_total,
            "phi_from_y0": phi_from_y0,
            "samples": samples,
            "adapted_for_test": {
                "num_children": len(K_final),
                "size_match_pass": size_match_pass,
                "meet_applicable_count": meet_count,
                "fallback_count": fallback_count
            },
            "policy": "multi-Y meet with fallback"
        }
    }

    return receipt
