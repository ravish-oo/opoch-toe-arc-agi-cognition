"""
WO-2: Receipts Schema + Writer

Provides helpers for writing receipts in JSONL format.
Each line is a complete JSON object representing one grid's receipt.

Per WO-1 spec: One JSONL line per grid (not per task).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_grid_receipt(
    task_id: str,
    grid_role: str,
    grid_index: int,
    H: int,
    W: int,
    sha256_T: str,
    sha256_T_again: str,
    codebook_sha256: str,
    type_sizes: Dict[int, int],
) -> Dict[str, Any]:
    """
    Build a WO-1 receipt dict for a single grid.

    Args:
        task_id: Task ID (e.g., "007bbfb7")
        grid_role: "train_output" or "test_input"
        grid_index: Index of the grid within the task (0, 1, 2, ...)
        H, W: Grid dimensions
        sha256_T: SHA256 hash of first Π computation
        sha256_T_again: SHA256 hash of second Π computation (on same input)
        codebook_sha256: SHA256 hash of the codebook
        type_sizes: Dict mapping type_id → count

    Returns:
        Complete receipt dict matching WO-1 spec
    """
    sum_sizes = sum(type_sizes.values())
    pass_idempotent = (sha256_T == sha256_T_again)
    pass_sum = (sum_sizes == H * W)

    return {
        "task_id": task_id,
        "grid_role": grid_role,
        "grid_index": grid_index,
        "H": H,
        "W": W,
        "sha256_T": sha256_T,
        "sha256_T_again": sha256_T_again,
        "pass_idempotent": pass_idempotent,
        "codebook_sha256": codebook_sha256,
        "type_sizes": type_sizes,
        "sum_sizes": sum_sizes,
        "pass_sum": pass_sum,
    }


class ReceiptWriter:
    """
    JSONL writer for receipts.

    Each receipt is written as a single line.
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = None

    def __enter__(self):
        self.file_handle = open(self.output_path, 'w', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()

    def write(self, receipt: Dict[str, Any]):
        """Write a single receipt as a JSON line."""
        if not self.file_handle:
            raise RuntimeError("ReceiptWriter not opened (use context manager)")

        self.file_handle.write(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
        self.file_handle.write('\n')
        self.file_handle.flush()


def read_receipts(receipts_path: Path) -> List[Dict[str, Any]]:
    """
    Read all receipts from a JSONL file.

    Args:
        receipts_path: Path to receipts.jsonl

    Returns:
        List of receipt dictionaries
    """
    receipts = []
    with open(receipts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                receipts.append(json.loads(line))
    return receipts


def find_receipt(
    receipts_path: Path,
    task_id: str,
    grid_role: str,
    grid_index: int,
) -> Optional[Dict[str, Any]]:
    """
    Find a specific grid's receipt.

    Args:
        receipts_path: Path to receipts.jsonl
        task_id: Task ID
        grid_role: "train_output" or "test_input"
        grid_index: Grid index

    Returns:
        Receipt dict if found, None otherwise
    """
    receipts = read_receipts(receipts_path)
    for receipt in receipts:
        if (
            receipt.get("task_id") == task_id
            and receipt.get("grid_role") == grid_role
            and receipt.get("grid_index") == grid_index
        ):
            return receipt
    return None
