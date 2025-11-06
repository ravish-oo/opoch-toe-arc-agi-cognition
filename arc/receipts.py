"""
WO-2: Receipts Skeleton + JSONL Writer

Provides helpers for SHA256 hashing and JSONL writing per WO-2 spec.
Receipts use consistent field names across WOs (sha256_T, sha256_T_again from WO-1).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def sha256_bytes(b: bytes) -> str:
    """
    Compute SHA256 hash of bytes.

    Args:
        b: Bytes to hash

    Returns:
        Hex digest string (64 characters)
    """
    return hashlib.sha256(b).hexdigest()


def sha256_ndarray(A: np.ndarray) -> str:
    """
    Compute deterministic SHA256 hash of a NumPy array.

    Converts array to int64 for deterministic byte representation across platforms.

    Args:
        A: NumPy array to hash

    Returns:
        Hex digest string (64 characters)
    """
    # Use int64 for deterministic cross-platform representation
    return hashlib.sha256(A.astype(np.int64).tobytes()).hexdigest()


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """
    Write records to a JSON Lines (JSONL) file.

    Per JSON Lines spec: one JSON object per line, UTF-8 encoded.
    https://jsonlines.org/

    Args:
        path: Output file path
        records: Iterable of dict-like records to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            # Compact JSON: no spaces after separators
            json_str = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
            f.write(json_str)
            f.write("\n")
