"""
WO-1: Π Ruler (Types)

Fixed, idempotent local observation that maps each pixel to a type ID via a 13-feature stencil.
Fully vectorized implementation using NumPy for deterministic, high-performance operation.

Axiom A0 (Truth): Π is task-agnostic, deterministic, and idempotent (Π² = Π).
"""

import hashlib
from typing import Dict, Tuple

import numpy as np


def types_from_output(Y: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, ...], int]]:
    """
    Compute the type mosaic T and codebook for a grid Y.

    For each pixel p=(r,c), extract a fixed 13-feature stencil φ(p):
      - 1 center: Y[r,c]
      - 4 neighbors: N, E, S, W
      - 4 two-step neighbors: 2N, 2E, 2S, 2W
      - 2 boundary flags: is_row_boundary, is_col_boundary
      - 2 parity: r mod 2, c mod 2

    Sentinel value -1 is used for out-of-bounds neighbors.

    Args:
        Y: Grid of shape (H, W) with integer color values

    Returns:
        T: Type mosaic of shape (H, W) with integer type IDs in [0, num_types-1]
        codebook: Dict mapping feature tuple → type ID (for receipts/debugging)

    Properties:
        - Fully vectorized (no Python loops over pixels)
        - Deterministic (same input → same output, same hashes)
        - Idempotent: types_from_output(T)[0] == T
    """
    Y = np.asarray(Y, dtype=np.int32)
    H, W = Y.shape

    # Pad with sentinel -1 for safe neighbor access (need 2-step = pad_width 2)
    Y_pad = np.pad(Y, pad_width=2, mode='constant', constant_values=-1)

    # Extract 13 channels by slicing the padded array
    # Index convention: Y_pad[2:2+H, 2:2+W] recovers original Y

    # Center (13 features total)
    c0_center = Y_pad[2:2+H, 2:2+W]           # (r, c)

    # 4-neighbors
    c1_N = Y_pad[1:1+H, 2:2+W]                 # (r-1, c)
    c2_E = Y_pad[2:2+H, 3:3+W]                 # (r, c+1)
    c3_S = Y_pad[3:3+H, 2:2+W]                 # (r+1, c)
    c4_W = Y_pad[2:2+H, 1:1+W]                 # (r, c-1)

    # 4 two-step neighbors
    c5_2N = Y_pad[0:0+H, 2:2+W]                # (r-2, c)
    c6_2E = Y_pad[2:2+H, 4:4+W]                # (r, c+2)
    c7_2S = Y_pad[4:4+H, 2:2+W]                # (r+2, c)
    c8_2W = Y_pad[2:2+H, 0:0+W]                # (r, c-2)

    # 2 boundary flags (broadcast to H x W)
    row_indices = np.arange(H)[:, None]
    col_indices = np.arange(W)[None, :]
    c9_row_boundary = np.broadcast_to(
        ((row_indices == 0) | (row_indices == H - 1)).astype(np.int32),
        (H, W)
    )
    c10_col_boundary = np.broadcast_to(
        ((col_indices == 0) | (col_indices == W - 1)).astype(np.int32),
        (H, W)
    )

    # 2 parity bits (broadcast to H x W)
    c11_row_parity = np.broadcast_to((row_indices & 1).astype(np.int32), (H, W))
    c12_col_parity = np.broadcast_to((col_indices & 1).astype(np.int32), (H, W))

    # Stack all 13 channels into shape (H, W, 13)
    feature_tensor = np.stack([
        c0_center,
        c1_N, c2_E, c3_S, c4_W,
        c5_2N, c6_2E, c7_2S, c8_2W,
        c9_row_boundary, c10_col_boundary,
        c11_row_parity, c12_col_parity
    ], axis=-1)

    # Reshape to (H*W, 13) for vectorized deduplication
    features_flat = feature_tensor.reshape(-1, 13)

    # Find unique feature vectors and assign type IDs
    # unique_features: (num_types, 13)
    # inverse: (H*W,) mapping each pixel to its type ID
    unique_features, inverse = np.unique(
        features_flat,
        axis=0,
        return_inverse=True
    )

    # Reshape inverse back to (H, W) to get type mosaic
    T = inverse.reshape(H, W).astype(np.int32)

    # Build codebook: feature tuple → type ID
    codebook = {
        tuple(feat): tid
        for tid, feat in enumerate(unique_features)
    }

    return T, codebook


def codebook_hash(codebook: Dict[Tuple[int, ...], int]) -> str:
    """
    Compute a deterministic SHA256 hash of the codebook.

    The codebook maps 13-feature tuples to type IDs. For determinism,
    we sort by type ID and hash the concatenated representation.

    Args:
        codebook: Dict mapping feature tuple → type ID

    Returns:
        Hex SHA256 hash string (64 chars)
    """
    # Sort by type ID for determinism
    sorted_items = sorted(codebook.items(), key=lambda x: x[1])

    # Serialize to bytes: for each (feature_tuple, type_id), write all values
    parts = []
    for feat_tuple, tid in sorted_items:
        parts.append(str(tid))
        parts.append(':')
        parts.append(','.join(map(str, feat_tuple)))
        parts.append(';')

    serialized = ''.join(parts).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def grid_hash(grid: np.ndarray) -> str:
    """
    Compute a deterministic SHA256 hash of a grid.

    Args:
        grid: NumPy array of shape (H, W)

    Returns:
        Hex SHA256 hash string (64 chars)
    """
    grid = np.asarray(grid, dtype=np.int32)
    # Use tobytes() for deterministic byte representation
    return hashlib.sha256(grid.tobytes()).hexdigest()


def compute_partition_sizes(T: np.ndarray, num_types: int) -> Dict[int, int]:
    """
    Compute |S_t| for each type t ∈ [0, num_types-1].

    Args:
        T: Type mosaic of shape (H, W)
        num_types: Total number of types

    Returns:
        Dict mapping type ID → count
    """
    unique, counts = np.unique(T, return_counts=True)
    partition = {int(t): int(c) for t, c in zip(unique, counts)}

    # Ensure all type IDs are present (even if count is 0)
    for t in range(num_types):
        if t not in partition:
            partition[t] = 0

    return partition
