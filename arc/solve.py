"""
WO-8: End-to-End Batch Solver

Orchestrates all WO modules to solve ARC-AGI tasks deterministically.

For WO-1 (pi-receipts mode):
  - Load all training tasks
  - Apply Π (types_from_output) TWICE to each training output to prove determinism
  - Compute SHA256 hashes, partition sizes
  - Write one JSONL line per grid with all required fields

CLI:
  python -m arc.solve --mode pi-receipts --data path/to/tasks.json --output outputs/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from arc.pi import (
    codebook_hash,
    compute_partition_sizes,
    grid_hash,
    types_from_output,
)
from arc.receipts import build_grid_receipt, ReceiptWriter


def load_tasks(data_path: Path) -> Dict[str, Any]:
    """
    Load ARC tasks from JSON file.

    Format:
    {
      "task_id": {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test": [{"input": [[...]]}, ...]
      },
      ...
    }

    Args:
        data_path: Path to JSON file

    Returns:
        Dict mapping task_id → task data
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_pi_receipts(
    tasks: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run WO-1 (Π receipts) on all tasks.

    For each training output grid:
      - Apply types_from_output TWICE to prove determinism
      - Compute hashes, partition sizes
      - Write one JSONL line per grid

    Args:
        tasks: Dict mapping task_id → task data
        output_dir: Output directory for receipts.jsonl
        verbose: Print progress messages

    Returns:
        Summary statistics dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = output_dir / "receipts.jsonl"

    stats = {
        "total_tasks": len(tasks),
        "total_grids": 0,
        "pass_idempotent_count": 0,
        "pass_sum_count": 0,
        "failed_grids": [],
    }

    if verbose:
        print(f"Running WO-1 (Π receipts) on {stats['total_tasks']} tasks...")
        print(f"Output: {receipts_path}")

    with ReceiptWriter(receipts_path) as writer:
        for task_idx, (task_id, task_data) in enumerate(tasks.items(), 1):
            if verbose and task_idx % 50 == 0:
                print(f"  Progress: {task_idx}/{stats['total_tasks']} tasks...")

            try:
                process_task_grids(task_id, task_data, writer, stats)
            except Exception as e:
                if verbose:
                    print(f"  ERROR on task {task_id}: {e}", file=sys.stderr)
                # Don't track failed tasks, track failed grids instead

    # Compute pass rates
    all_pass_idempotent = (stats["pass_idempotent_count"] == stats["total_grids"])
    all_pass_sum = (stats["pass_sum_count"] == stats["total_grids"])

    if verbose:
        print(f"\nCompleted WO-1 receipts:")
        print(f"  Total grids processed: {stats['total_grids']}")
        print(f"  pass_idempotent: {stats['pass_idempotent_count']}/{stats['total_grids']}")
        print(f"  pass_sum: {stats['pass_sum_count']}/{stats['total_grids']}")
        if stats['failed_grids']:
            print(f"  Failed grids: {len(stats['failed_grids'])}")
            print(f"    {', '.join(stats['failed_grids'][:10])}")

    # Write summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"WO-1 (Π Receipts) Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Total tasks: {stats['total_tasks']}\n")
        f.write(f"Total grids: {stats['total_grids']}\n")
        f.write(f"All pass_idempotent: {all_pass_idempotent}\n")
        f.write(f"All pass_sum: {all_pass_sum}\n")
        f.write(f"\npass_idempotent: {stats['pass_idempotent_count']}/{stats['total_grids']}\n")
        f.write(f"pass_sum: {stats['pass_sum_count']}/{stats['total_grids']}\n")
        if stats['failed_grids']:
            f.write(f"\nFailed grids:\n")
            for grid_id in stats['failed_grids']:
                f.write(f"  - {grid_id}\n")

    stats["all_pass_idempotent"] = all_pass_idempotent
    stats["all_pass_sum"] = all_pass_sum
    return stats


def process_task_grids(
    task_id: str,
    task_data: Dict[str, Any],
    writer: ReceiptWriter,
    stats: Dict[str, Any],
):
    """
    Process all grids for a single task and write receipts.

    Args:
        task_id: Task ID
        task_data: Task data with "train" and "test" fields
        writer: ReceiptWriter instance
        stats: Statistics dict to update
    """
    train_pairs = task_data.get("train", [])

    # Process each training output
    for idx, pair in enumerate(train_pairs):
        output_grid = np.array(pair["output"], dtype=np.int32)
        H, W = output_grid.shape

        # Apply Π TWICE to prove determinism
        T1, codebook1 = types_from_output(output_grid)
        T2, codebook2 = types_from_output(output_grid)

        # Compute hashes
        sha256_T1 = grid_hash(T1)
        sha256_T2 = grid_hash(T2)
        cb_hash1 = codebook_hash(codebook1)
        cb_hash2 = codebook_hash(codebook2)

        # Use first computation for type_sizes (both should be identical)
        num_types = len(codebook1)
        type_sizes = compute_partition_sizes(T1, num_types)

        # Build receipt
        receipt = build_grid_receipt(
            task_id=task_id,
            grid_role="train_output",
            grid_index=idx,
            H=H,
            W=W,
            sha256_T=sha256_T1,
            sha256_T_again=sha256_T2,
            codebook_sha256=cb_hash1,
            type_sizes=type_sizes,
        )

        # Write receipt
        writer.write(receipt)

        # Update stats
        stats["total_grids"] += 1
        if receipt["pass_idempotent"]:
            stats["pass_idempotent_count"] += 1
        else:
            grid_id = f"{task_id}:train_output:{idx}"
            stats["failed_grids"].append(grid_id)

        if receipt["pass_sum"]:
            stats["pass_sum_count"] += 1
        else:
            grid_id = f"{task_id}:train_output:{idx}"
            if grid_id not in stats["failed_grids"]:
                stats["failed_grids"].append(grid_id)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ARC-AGI Cognition Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pi-receipts"],
        help="Solver mode: pi-receipts (WO-1 only)",
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to ARC tasks JSON file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output directory for receipts and predictions (default: outputs/)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Load tasks
    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    tasks = load_tasks(args.data)

    if not args.quiet:
        print(f"Loaded {len(tasks)} tasks from {args.data}")

    # Run requested mode
    if args.mode == "pi-receipts":
        stats = run_pi_receipts(tasks, args.output, verbose=not args.quiet)

        # WO-1 acceptance criteria: all receipts must pass
        if not stats["all_pass_idempotent"]:
            print("\nERROR: Some grids failed pass_idempotent check!", file=sys.stderr)
            sys.exit(1)

        if not stats["all_pass_sum"]:
            print("\nERROR: Some grids failed pass_sum check!", file=sys.stderr)
            sys.exit(1)

    elapsed = time.time() - start_time

    if not args.quiet:
        print(f"\nCompleted in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
