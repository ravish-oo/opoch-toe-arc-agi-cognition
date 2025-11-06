"""
Harness + Receipts Runner (WO-2, WO-3A, WO-3B, WO-3C)

Deterministic corpus runner that loads ARC JSON and emits receipts.

Modes:
  - pi-receipts: Π receipts for training outputs (WO-2)
  - free-simple-receipts: Simple FREE verifiers at color level (WO-3A)
  - free-tile-receipts: Types-periodic tile verifier (WO-3B)
  - free-sbs-y-receipts: SBS-Y verifier on types (WO-3C)

CLI:
  python -m arc.solve --mode pi-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-simple-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-tile-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-sbs-y-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from arc.pi import types_from_output, codebook_hash, compute_partition_sizes
from arc.receipts import sha256_ndarray, write_jsonl
from arc.free_simple import (
    verify_simple_free, get_detailed_checks,
    verify_tile_types, get_tile_detailed_checks
)
from arc.free_sbs import (
    verify_SBS_Y, get_sbs_y_detailed_checks
)


def load_tasks_from_json(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load ARC tasks from JSON file.

    Returns canonicalized dict mapping task_id -> payload with:
      - payload["train"]: list of dicts with "input" and "output" arrays
      - payload["test"]: list of dicts with "input" array

    Args:
        path: Path to challenges JSON file

    Returns:
        Dict mapping task_id -> task payload
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_pi_receipts(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
    include_test_pi: bool = False,
) -> None:
    """
    Run WO-2 Π receipts mode.

    For each task:
      - Process each training output grid
      - Optionally process test input grids (if include_test_pi=True)
      - Compute Π twice to verify idempotence
      - Write one JSONL line per grid

    Args:
        tasks: Dict mapping task_id -> task data
        out_path: Output path for receipts JSONL file
        include_test_pi: Whether to include test input receipts
    """
    receipts: List[Dict[str, Any]] = []

    # Counters for summary
    total_tasks = len(tasks)
    total_grids = 0
    idempotence_fail = 0
    sum_fail = 0

    for task_id, task_data in tasks.items():
        # Process training outputs
        train_pairs = task_data.get("train", [])
        for grid_index, pair in enumerate(train_pairs):
            Y = np.array(pair["output"], dtype=np.int32)
            receipt = _compute_pi_receipt(
                task_id=task_id,
                grid_role="train_output",
                grid_index=grid_index,
                grid=Y,
            )
            receipts.append(receipt)
            total_grids += 1

            if not receipt["pass_idempotent"]:
                idempotence_fail += 1
            if not receipt["pass_sum"]:
                sum_fail += 1

        # Optionally process test inputs
        if include_test_pi:
            test_pairs = task_data.get("test", [])
            for grid_index, pair in enumerate(test_pairs):
                X = np.array(pair["input"], dtype=np.int32)
                receipt = _compute_pi_receipt(
                    task_id=task_id,
                    grid_role="test_input",
                    grid_index=grid_index,
                    grid=X,
                )
                receipts.append(receipt)
                total_grids += 1

                if not receipt["pass_idempotent"]:
                    idempotence_fail += 1
                if not receipt["pass_sum"]:
                    sum_fail += 1

    # Write all receipts to JSONL
    write_jsonl(out_path, receipts)

    # Log summary (single INFO line per WO-2 spec)
    logging.info(
        f"Processed tasks={total_tasks}, grids={total_grids}, "
        f"idempotence_fail={idempotence_fail}, sum_fail={sum_fail}"
    )


def _compute_pi_receipt(
    task_id: str,
    grid_role: str,
    grid_index: int,
    grid: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute Π receipt for a single grid.

    Applies types_from_output twice to verify idempotence.

    Args:
        task_id: Task ID
        grid_role: "train_output" or "test_input"
        grid_index: Grid index within task
        grid: NumPy array (H, W)

    Returns:
        Receipt dict with all required fields
    """
    H, W = grid.shape

    # Apply Π twice to check idempotence
    T1, codebook1 = types_from_output(grid)
    T2, codebook2 = types_from_output(grid)

    # Compute hashes
    sha256_T_once = sha256_ndarray(T1)
    sha256_T_twice = sha256_ndarray(T2)
    cb_hash = codebook_hash(codebook1)

    # Compute partition
    num_types = len(codebook1)
    type_sizes = compute_partition_sizes(T1, num_types)
    sum_sizes = sum(type_sizes.values())

    # Validation checks
    pass_idempotent = (sha256_T_once == sha256_T_twice)
    pass_sum = (sum_sizes == H * W)

    # Build receipt with WO-1 field names (sha256_T, sha256_T_again)
    return {
        "task_id": task_id,
        "grid_role": grid_role,
        "grid_index": grid_index,
        "H": H,
        "W": W,
        "sha256_T": sha256_T_once,
        "sha256_T_again": sha256_T_twice,
        "pass_idempotent": pass_idempotent,
        "codebook_sha256": cb_hash,
        "type_sizes": type_sizes,
        "sum_sizes": sum_sizes,
        "pass_sum": pass_sum,
    }


def run_free_simple_receipts(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Run WO-3A simple FREE verifiers mode.

    For each task:
      - Process each training pair (X->Y)
      - Verify simple FREE candidates at color level
      - Write one record per pair + one union record per task

    Args:
        tasks: Dict mapping task_id -> task data
        out_path: Output path for receipts JSONL file
    """
    receipts: List[Dict[str, Any]] = []

    # Counters for summary
    total_tasks = len(tasks)
    total_pairs = 0

    for task_id, task_data in tasks.items():
        train_pairs = task_data.get("train", [])
        task_union_kinds: List[str] = []

        # Process each training pair
        for pair_index, pair in enumerate(train_pairs):
            X = np.array(pair["input"], dtype=np.int32)
            Y = np.array(pair["output"], dtype=np.int32)

            # Verify simple FREE candidates
            candidates = verify_simple_free(X, Y)

            # Get detailed checks for receipt
            detailed_checks = get_detailed_checks(X, Y)

            # Build per-pair receipt
            pair_receipt = {
                "task_id": task_id,
                "pair_index": pair_index,
                "free_simple": detailed_checks,
                "candidates": [[kind, params] for kind, params in candidates],
            }

            receipts.append(pair_receipt)
            total_pairs += 1

            # Collect unique kinds for task union
            for kind, _ in candidates:
                if kind not in task_union_kinds:
                    task_union_kinds.append(kind)

        # After all pairs, add task-level union record
        task_union_receipt = {
            "task_id": task_id,
            "free_simple_union": task_union_kinds,
        }
        receipts.append(task_union_receipt)

    # Write all receipts to JSONL
    write_jsonl(out_path, receipts)

    # Log summary
    logging.info(
        f"Processed tasks={total_tasks}, pairs={total_pairs}"
    )


def run_free_tile_receipts(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Run WO-3B types-periodic tile verifier mode.

    For each task:
      - Process each training pair (X->Y)
      - Compute T_Y = Π(Y) type mosaic
      - Verify tile FREE candidate on types
      - Write one record per pair + one union record per task

    Args:
        tasks: Dict mapping task_id -> task data
        out_path: Output path for receipts JSONL file
    """
    receipts: List[Dict[str, Any]] = []

    # Counters for summary
    total_tasks = len(tasks)
    total_pairs = 0
    tile_match_count = 0

    for task_id, task_data in tasks.items():
        train_pairs = task_data.get("train", [])
        task_has_tile = False

        # Process each training pair
        for pair_index, pair in enumerate(train_pairs):
            X = np.array(pair["input"], dtype=np.int32)
            Y = np.array(pair["output"], dtype=np.int32)

            # Compute type mosaic T_Y from Π(Y)
            T_Y, _ = types_from_output(Y)

            # Verify tile on types
            candidate = verify_tile_types(X, Y, T_Y)

            # Get detailed checks for receipt
            detailed_checks = get_tile_detailed_checks(X, Y, T_Y)

            # Build per-pair receipt
            pair_receipt = {
                "task_id": task_id,
                "pair_index": pair_index,
                "free_tile_types": detailed_checks,
            }

            # Add candidate field only if verification succeeded
            if candidate is not None:
                kind, (sh, sw) = candidate
                pair_receipt["candidate"] = [kind, [sh, sw]]
                tile_match_count += 1
                task_has_tile = True

            receipts.append(pair_receipt)
            total_pairs += 1

        # After all pairs, add task-level union record
        task_union_receipt = {
            "task_id": task_id,
            "free_tile_union": ["tile"] if task_has_tile else [],
        }
        receipts.append(task_union_receipt)

    # Write all receipts to JSONL
    write_jsonl(out_path, receipts)

    # Log summary
    logging.info(
        f"Processed tasks={total_tasks}, pairs={total_pairs}, tile_matches={tile_match_count}"
    )


def run_free_sbs_y_receipts(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Run WO-3C SBS-Y verifier mode.

    For each task:
      - Process each training pair (X->Y)
      - Compute T_Y = Π(Y) type mosaic
      - Verify SBS-Y candidate on types
      - Write one record per pair

    Args:
        tasks: Dict mapping task_id -> task data
        out_path: Output path for receipts JSONL file
    """
    receipts: List[Dict[str, Any]] = []

    # Counters for summary
    total_tasks = len(tasks)
    total_pairs = 0
    sbs_y_match_count = 0

    for task_id, task_data in tasks.items():
        train_pairs = task_data.get("train", [])

        # Process each training pair
        for pair_index, pair in enumerate(train_pairs):
            X = np.array(pair["input"], dtype=np.int32)
            Y = np.array(pair["output"], dtype=np.int32)

            # Compute type mosaic T_Y from Π(Y)
            T_Y, _ = types_from_output(Y)

            # Verify SBS-Y on types
            candidate = verify_SBS_Y(X, T_Y)

            # Get detailed checks for receipt
            detailed_checks = get_sbs_y_detailed_checks(X, T_Y)

            # Build per-pair receipt
            pair_receipt = {
                "task_id": task_id,
                "pair_index": pair_index,
                "free_sbs_y": detailed_checks,
            }

            # Add candidate field only if verification succeeded
            if candidate is not None:
                kind, (sh, sw, sigma_table, template_hashes) = candidate
                pair_receipt["candidate"] = [kind, [sh, sw]]
                sbs_y_match_count += 1

            receipts.append(pair_receipt)
            total_pairs += 1

    # Write all receipts to JSONL
    write_jsonl(out_path, receipts)

    # Log summary
    logging.info(
        f"Processed tasks={total_tasks}, pairs={total_pairs}, sbs_y_matches={sbs_y_match_count}"
    )


def main() -> None:
    """CLI entry point with argparse."""
    # Configure logging (single INFO line per WO-2)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    parser = argparse.ArgumentParser(
        description="ARC-AGI Cognition Solver - WO-2 Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pi-receipts", "free-simple-receipts", "free-tile-receipts", "free-sbs-y-receipts"],
        help="Solver mode: pi-receipts (WO-2), free-simple-receipts (WO-3A), free-tile-receipts (WO-3B), or free-sbs-y-receipts (WO-3C)",
    )

    parser.add_argument(
        "--challenges",
        type=Path,
        required=True,
        help="Path to ARC challenges JSON file",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for receipts JSONL file",
    )

    parser.add_argument(
        "--include-test-pi",
        action="store_true",
        help="Include test input grids in Π receipts (for sanity checking)",
    )

    args = parser.parse_args()

    # Load tasks
    if not args.challenges.exists():
        logging.error(f"Challenges file not found: {args.challenges}")
        raise SystemExit(1)

    tasks = load_tasks_from_json(args.challenges)

    # Run requested mode
    if args.mode == "pi-receipts":
        run_pi_receipts(
            tasks=tasks,
            out_path=args.out,
            include_test_pi=args.include_test_pi,
        )
    elif args.mode == "free-simple-receipts":
        run_free_simple_receipts(
            tasks=tasks,
            out_path=args.out,
        )
    elif args.mode == "free-tile-receipts":
        run_free_tile_receipts(
            tasks=tasks,
            out_path=args.out,
        )
    elif args.mode == "free-sbs-y-receipts":
        run_free_sbs_y_receipts(
            tasks=tasks,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
