"""
WO-6: Quotas K (Paid) + Y₀ Selection

Deterministically select one training output (Y₀) to read quotas from.
Compute per-type color quotas K_{S,c} on Π(Y₀).
Emit receipts proving selection policy and validity.

Critical: Quotas come ONLY from training outputs, NEVER from inputs.
"""

import numpy as np
from typing import Dict, Any, Tuple

from arc.pi import types_from_output


def choose_Y0(task: Dict[str, Any]) -> int:
    """
    Deterministically choose Y₀ by palette matching.

    Policy:
      1. Get test input nonzero palette P_test
      2. For each training output Y_i, get nonzero palette P_i
      3. Return first i where P_i == P_test
      4. Fallback to 0 if no match

    Args:
        task: Task dict with "train" and "test" keys

    Returns:
        y0_index: Index of selected training output
    """
    # Get test input
    X_test = np.array(task["test"][0]["input"], dtype=np.int32)
    P_test = set(np.unique(X_test).tolist()) - {0}

    # Try to match training outputs
    for i, pair in enumerate(task["train"]):
        Y_i = np.array(pair["output"], dtype=np.int32)
        P_i = set(np.unique(Y_i).tolist()) - {0}

        if P_i == P_test:
            return i

    # Fallback
    return 0


def quotas(Y0: np.ndarray, T0: np.ndarray, C: int = 10) -> Dict[int, np.ndarray]:
    """
    Compute per-type color quotas.

    For each type S in Π(Y₀), count how many pixels of each color c appear:
      K[S][c] = count of pixels with color c in type S

    Conservation law: ∑_c K[S,c] = |S|

    Args:
        Y0: Training output (selected Y₀)
        T0: Type mosaic Π(Y₀)
        C: Number of colors (default 10 for ARC)

    Returns:
        Dict mapping type_id -> length-C vector of color counts
    """
    Y0 = np.asarray(Y0, dtype=np.int32)
    T0 = np.asarray(T0, dtype=np.int32)

    # Get unique type IDs
    unique_types = np.unique(T0).tolist()

    # Compute quotas for each type
    K = {}
    for type_id in unique_types:
        # Isolate pixels in this type
        mask = (T0 == type_id)
        colors = Y0[mask]

        # Count colors with bincount (guaranteed length-C)
        counts = np.bincount(colors, minlength=C)[:C]
        K[type_id] = counts

    return K


def generate_quotas_receipt(task: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Generate quotas receipt for a task.

    Args:
        task: Task dict with "train" and "test" keys
        task_id: Task identifier

    Returns:
        Receipt dict ready for JSONL output
    """
    C = 10

    # Choose Y₀
    y0_index = choose_Y0(task)

    # Get Y₀ and compute Π(Y₀)
    Y0 = np.array(task["train"][y0_index]["output"], dtype=np.int32)
    T0, _ = types_from_output(Y0)

    # Compute quotas
    K = quotas(Y0, T0, C)

    # Determine y0_reason by re-checking palette match
    X_test = np.array(task["test"][0]["input"], dtype=np.int32)
    P_test = set(np.unique(X_test).tolist()) - {0}
    P_y0 = set(np.unique(Y0).tolist()) - {0}
    y0_reason = "palette_match" if P_y0 == P_test else "fallback_first"

    # Get test input palette
    palette_test_nonzero = sorted(P_test)

    # Get ALL training output palettes (list of lists)
    palette_train_nonzero = []
    for pair in task["train"]:
        Y_i = np.array(pair["output"], dtype=np.int32)
        P_i = sorted((set(np.unique(Y_i).tolist()) - {0}))
        palette_train_nonzero.append(P_i)

    # Convert K to serializable format (str keys, list values)
    K_serializable = {str(type_id): counts.tolist() for type_id, counts in K.items()}

    # Compute sum_checks (verify conservation law)
    sum_checks = {}
    for type_id, counts in K.items():
        # Count pixels in this type
        size = int(np.sum(T0 == type_id))
        sum_val = int(np.sum(counts))
        sum_checks[str(type_id)] = {
            "size": size,
            "sum": sum_val,
            "pass": (size == sum_val)
        }

    # Build receipt
    receipt = {
        "task_id": task_id,
        "quotas": {
            "C": C,
            "y0_index": y0_index,
            "y0_reason": y0_reason,
            "palette_test_nonzero": palette_test_nonzero,
            "palette_train_nonzero": palette_train_nonzero,
            "K": K_serializable,
            "sum_checks": sum_checks
        }
    }

    return receipt
