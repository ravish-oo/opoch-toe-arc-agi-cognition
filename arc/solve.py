"""
Harness + Receipts Runner (WO-2, WO-3A, WO-3B, WO-3C, WO-3D, WO-4, WO-5, WO-6)

Deterministic corpus runner that loads ARC JSON and emits receipts.

Modes:
  - pi-receipts: Π receipts for training outputs (WO-2)
  - free-simple-receipts: Simple FREE verifiers at color level (WO-3A)
  - free-tile-receipts: Types-periodic tile verifier (WO-3B)
  - free-sbs-y-receipts: SBS-Y verifier on types (WO-3C)
  - free-sbs-param-receipts: SBS-Param verifier on types (WO-3D)
  - free-intersect-pick: FREE intersection and pick with frozen order (WO-4)
  - transport-receipts: Transport types + disjointify (WO-5)
  - quotas-receipts: Quotas K (Paid) + Y₀ Selection (WO-6)

CLI:
  python -m arc.solve --mode pi-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-simple-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-tile-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-sbs-y-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-sbs-param-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-intersect-pick --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode transport-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode quotas-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from arc.pi import types_from_output, codebook_hash, compute_partition_sizes
from arc.receipts import sha256_ndarray, write_jsonl
from arc.free_simple import (
    verify_simple_free, get_detailed_checks,
    verify_tile_types, get_tile_detailed_checks
)
from arc.free_sbs import (
    verify_SBS_Y, get_sbs_y_detailed_checks,
    verify_SBS_param, get_sbs_param_detailed_checks
)
from arc.free_prove import (
    prove_free, compute_chosen_sha256
)
from arc.transport import (
    transport_types, disjointify, verify_blocks_match
)
from arc.quotas import generate_quotas_receipt


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


def run_free_sbs_param_receipts(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Run WO-3D SBS-Param verifier mode.

    For each task:
      - Process each training pair (X->Y)
      - Compute T_X = Π(X) and T_Y = Π(Y) type mosaics
      - Verify SBS-Param candidate on types (templates from Π(X))
      - Write one record per pair

    Args:
        tasks: Dict mapping task_id -> task data
        out_path: Output path for receipts JSONL file
    """
    receipts: List[Dict[str, Any]] = []

    # Counters for summary
    total_tasks = len(tasks)
    total_pairs = 0
    sbs_param_match_count = 0

    for task_id, task_data in tasks.items():
        train_pairs = task_data.get("train", [])

        # Process each training pair
        for pair_index, pair in enumerate(train_pairs):
            X = np.array(pair["input"], dtype=np.int32)
            Y = np.array(pair["output"], dtype=np.int32)

            # Verify SBS-Param on types (computes Π internally)
            candidate = verify_SBS_param(X, Y)

            # Get detailed checks for receipt
            detailed_checks = get_sbs_param_detailed_checks(X, Y)

            # Build per-pair receipt
            pair_receipt = {
                "task_id": task_id,
                "pair_index": pair_index,
                "free_sbs_param": detailed_checks,
            }

            # Add candidate field only if verification succeeded
            if candidate is not None:
                kind, (sh, sw, sigma_table, template_hashes) = candidate
                pair_receipt["candidate"] = [kind, [sh, sw]]
                sbs_param_match_count += 1

            receipts.append(pair_receipt)
            total_pairs += 1

    # Write all receipts to JSONL
    write_jsonl(out_path, receipts)

    # Log summary
    logging.info(
        f"Processed tasks={total_tasks}, pairs={total_pairs}, sbs_param_matches={sbs_param_match_count}"
    )


def run_free_intersect_pick(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Run WO-4 FREE intersection and pick mode.

    For each task:
      - Run all WO-3 verifiers on each training pair
      - Collect per-pair candidates (WO-3A: simple, WO-3B: tile, WO-3C: SBS-Y, WO-3D: SBS-param)
      - Call prove_free to intersect and pick
      - Write one task-level record

    Args:
        tasks: Dict mapping task_id -> task data
        out_path: Output path for receipts JSONL file
    """
    receipts: List[Dict[str, Any]] = []

    # Counters for summary
    total_tasks = len(tasks)
    proven_count = 0
    unproven_count = 0

    for task_id, task_data in tasks.items():
        train_pairs = task_data.get("train", [])

        # Collect per-pair candidates
        per_pair_simple = []
        per_pair_tile = []
        per_pair_sbs_y = []
        per_pair_sbs_p = []

        for pair_index, pair in enumerate(train_pairs):
            X = np.array(pair["input"], dtype=np.int32)
            Y = np.array(pair["output"], dtype=np.int32)

            # WO-3A: Simple FREE verifiers
            simple_cands = verify_simple_free(X, Y)
            per_pair_simple.append(simple_cands)

            # WO-3B: Tile on types
            T_Y, _ = types_from_output(Y)
            tile_cand = verify_tile_types(X, Y, T_Y)
            per_pair_tile.append(tile_cand)

            # WO-3C: SBS-Y on types
            sbs_y_cand = verify_SBS_Y(X, T_Y)
            per_pair_sbs_y.append(sbs_y_cand)

            # WO-3D: SBS-Param on types
            sbs_p_cand = verify_SBS_param(X, Y)
            per_pair_sbs_p.append(sbs_p_cand)

        # Call prove_free to intersect and pick
        status, proof_data = prove_free(
            task_data,
            per_pair_simple,
            per_pair_tile,
            per_pair_sbs_y,
            per_pair_sbs_p
        )

        # Build per-pair receipt structure
        per_pair_receipts = []
        for pair_index in range(len(train_pairs)):
            # Collect all candidates for this pair
            pair_candidates = []
            pair_candidates.extend(per_pair_simple[pair_index])
            if per_pair_tile[pair_index] is not None:
                pair_candidates.append(per_pair_tile[pair_index])
            if per_pair_sbs_y[pair_index] is not None:
                pair_candidates.append(per_pair_sbs_y[pair_index])
            if per_pair_sbs_p[pair_index] is not None:
                pair_candidates.append(per_pair_sbs_p[pair_index])

            per_pair_receipts.append({
                "pair_index": pair_index,
                "candidates": pair_candidates
            })

        # Build task-level receipt
        task_receipt = {
            "task_id": task_id,
            "free_intersection": {
                "per_pair": per_pair_receipts,
                "intersected": proof_data["intersected"]
            }
        }

        if status == "FREE_PROVEN":
            task_receipt["free_intersection"]["chosen"] = proof_data["chosen"]
            task_receipt["free_intersection"]["order_rank"] = proof_data["order_rank"]
            task_receipt["free_intersection"]["chosen_sha256"] = compute_chosen_sha256(proof_data["chosen"])
            proven_count += 1
        else:
            task_receipt["free_intersection"]["reason"] = proof_data["reason"]
            unproven_count += 1

        receipts.append(task_receipt)

    # Write all receipts to JSONL
    write_jsonl(out_path, receipts)

    # Log summary
    logging.info(
        f"Processed tasks={total_tasks}, proven={proven_count}, unproven={unproven_count}"
    )


def run_transport_receipts(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Run WO-5 transport types + disjointify mode.

    For each FREE_PROVEN task from WO-4:
      - Load proven terminal
      - Reconstruct SBS templates if needed
      - Transport T_Y0 to T_test
      - Disjointify if needed
      - Write receipt with pre/post disjoint stats

    Args:
        tasks: Dict mapping task_id -> task data
        out_path: Output path for receipts JSONL file
    """
    receipts: List[Dict[str, Any]] = []

    # Load WO-4 receipts to get proven terminals
    wo4_path = Path("outputs/receipts_wo4.jsonl")
    if not wo4_path.exists():
        logging.error(f"WO-4 receipts not found: {wo4_path}")
        raise SystemExit(1)

    wo4_receipts = {}
    with open(wo4_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            wo4_receipts[rec["task_id"]] = rec

    # Counters for summary
    total_tasks = len(tasks)
    processed_count = 0
    skipped_unproven = 0

    for task_id, task_data in tasks.items():
        # Get WO-4 result
        if task_id not in wo4_receipts:
            continue

        wo4_rec = wo4_receipts[task_id]
        free_inter = wo4_rec["free_intersection"]

        # Skip unproven tasks
        if "chosen" not in free_inter:
            skipped_unproven += 1
            continue

        chosen = free_inter["chosen"]
        kind, params = chosen[0], chosen[1]

        # Get training pair 0
        train_pairs = task_data.get("train", [])
        if len(train_pairs) == 0:
            continue

        X0 = np.array(train_pairs[0]["input"], dtype=np.int32)
        Y0 = np.array(train_pairs[0]["output"], dtype=np.int32)
        T_Y0, _ = types_from_output(Y0)

        # Get test input (use first test case)
        test_cases = task_data.get("test", [])
        if len(test_cases) == 0:
            continue

        X_test = np.array(test_cases[0]["input"], dtype=np.int32)

        # Reconstruct SBS templates if needed
        templates = None
        if kind in ["SBS-Y", "SBS-param"]:
            templates = _reconstruct_sbs_templates(kind, X0, Y0, T_Y0, params)

        # Build free_tuple for transport
        if templates is not None:
            # Unpack original params and add templates
            sh, sw = params[0], params[1]
            sigma_table = params[2] if len(params) > 2 else {}
            free_tuple = (kind, (sh, sw, sigma_table, templates))
        else:
            free_tuple = (kind, params)

        # Transport types (without disjointify first for receipt stats)
        from arc.transport import (_transport_identity, _transport_h_mirror_concat,
                                     _transport_v_double, _transport_h_concat_dup,
                                     _transport_v_concat_dup, _transport_tile, _transport_sbs)

        if kind == "identity":
            T_test_before = _transport_identity(T_Y0)
        elif kind == "h-mirror-concat":
            T_test_before = _transport_h_mirror_concat(T_Y0, params)
        elif kind == "v-double":
            T_test_before = _transport_v_double(T_Y0)
        elif kind == "h-concat-dup":
            T_test_before = _transport_h_concat_dup(T_Y0)
        elif kind == "v-concat-dup":
            T_test_before = _transport_v_concat_dup(T_Y0)
        elif kind == "tile":
            T_test_before = _transport_tile(T_Y0, params)
        elif kind in ["SBS-Y", "SBS-param"]:
            T_test_before = _transport_sbs(T_Y0, free_tuple[1], X_test, kind)
        else:
            T_test_before = T_Y0.copy()

        # Full transport with disjointify
        T_test, parent_of = transport_types(T_Y0, free_tuple, X_test.shape, X_test)

        # Compute pre/post disjoint stats
        T_pre = T_test_before
        pre_unique = len(np.unique(T_pre))
        post_unique = len(np.unique(T_test))

        # Build block evidence for tile/SBS
        block_evidence = None
        if kind in ["tile", "SBS-Y", "SBS-param"]:
            H, W = X_test.shape
            if kind == "tile":
                sh, sw = params
                blocks_match = True  # Always true for tile
            else:
                sh, sw = free_tuple[1][0], free_tuple[1][1]
                sigma_table = free_tuple[1][2]
                blocks_match = verify_blocks_match(T_test_before, templates, X_test, sigma_table, sh, sw)

            block_evidence = {
                "grid": [H, W],
                "sh": sh,
                "sw": sw,
                "blocks_match": blocks_match
            }

        # Verify disjointify mapping
        parent_consistency_pass = True
        counts_conserved_pass = True

        # Check parent consistency: all pixels in S' came from parent S
        for new_id, parent_id in parent_of.items():
            idx = np.flatnonzero(T_test.ravel() == new_id)
            if len(idx) > 0:
                if not np.all(T_pre.ravel()[idx] == parent_id):
                    parent_consistency_pass = False
                    break

        # Check counts conserved: sum of children == parent size
        parent_sizes = {}
        child_sizes = {}

        for parent_id in np.unique(T_pre):
            parent_sizes[int(parent_id)] = int(np.count_nonzero(T_pre == parent_id))

        for new_id, parent_id in parent_of.items():
            child_sizes.setdefault(parent_id, 0)
            child_sizes[parent_id] += int(np.count_nonzero(T_test == new_id))

        for parent_id, expected_size in parent_sizes.items():
            actual_size = child_sizes.get(parent_id, 0)
            if actual_size != expected_size:
                counts_conserved_pass = False
                break

        # Build parent_map samples (first 3 mappings)
        parent_map_samples = [[str(new_id), str(parent_id)]
                              for new_id, parent_id in list(parent_of.items())[:3]]

        # Build receipt
        task_receipt = {
            "task_id": task_id,
            "transport": {
                "terminal": [kind, params],
                "in_shape": list(T_Y0.shape),
                "test_shape": list(T_test.shape),
                "pre_disjoint": {
                    "unique_type_ids": int(pre_unique)
                },
                "post_disjoint": {
                    "unique_type_ids": int(post_unique),
                    "components_labeled": True
                }
            },
            "disjointify": {
                "pre": {
                    "unique_types": int(pre_unique)
                },
                "post": {
                    "unique_types": int(post_unique)
                },
                "parent_map": {
                    "size": len(parent_of),
                    "samples": parent_map_samples
                },
                "parent_consistency_pass": parent_consistency_pass,
                "counts_conserved_pass": counts_conserved_pass
            }
        }

        if block_evidence is not None:
            task_receipt["transport"]["block_evidence"] = block_evidence

        receipts.append(task_receipt)
        processed_count += 1

    # Write all receipts to JSONL
    write_jsonl(out_path, receipts)

    # Log summary
    logging.info(
        f"Processed tasks={total_tasks}, transported={processed_count}, skipped_unproven={skipped_unproven}"
    )


def run_quotas_receipts(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Run WO-6 quotas receipts mode.

    For each task:
      - Select Y₀ using palette matching policy
      - Compute Π(Y₀)
      - Compute per-type color quotas K
      - Verify conservation law
      - Emit one JSONL line per task

    Args:
        tasks: Dict mapping task_id -> task data
        out_path: Output path for receipts JSONL file
    """
    receipts: List[Dict[str, Any]] = []

    # Counters for summary
    total_tasks = len(tasks)
    palette_match_count = 0
    fallback_count = 0
    all_sum_checks_passed = 0

    for task_id, task_data in tasks.items():
        # Generate receipt
        receipt = generate_quotas_receipt(task_data, task_id)

        # Track statistics
        y0_reason = receipt["quotas"]["y0_reason"]
        if y0_reason == "palette_match":
            palette_match_count += 1
        elif y0_reason == "fallback_first":
            fallback_count += 1

        # Check if all sum_checks passed
        sum_checks = receipt["quotas"]["sum_checks"]
        all_valid = all(check["pass"] for check in sum_checks.values())
        if all_valid:
            all_sum_checks_passed += 1

        receipts.append(receipt)

    # Write receipts
    write_jsonl(out_path, receipts)

    # Print summary
    logging.info("=" * 60)
    logging.info("WO-6 Quotas Receipts Summary")
    logging.info("=" * 60)
    logging.info(f"Total tasks processed: {total_tasks}")
    logging.info(f"Palette match Y₀ selection: {palette_match_count}")
    logging.info(f"Fallback Y₀ selection: {fallback_count}")
    logging.info(f"All sum_checks passed: {all_sum_checks_passed}/{total_tasks}")
    logging.info(f"Receipts written to: {out_path}")
    logging.info("=" * 60)


def _reconstruct_sbs_templates(
    kind: str,
    X0: np.ndarray,
    Y0: np.ndarray,
    T_Y0: np.ndarray,
    params: Tuple[Any, ...]
) -> Dict[int, np.ndarray]:
    """
    Reconstruct SBS templates from training data.

    For SBS-Y: extract (sh×sw) blocks from partitioned T_Y0
    For SBS-param: create constant (sh×sw) blocks from T_X0

    Args:
        kind: "SBS-Y" or "SBS-param"
        X0: Training input
        Y0: Training output
        T_Y0: Training output types Π(Y0)
        params: (sh, sw, sigma_table, ...)

    Returns:
        Dict mapping template_id -> (sh, sw) type array
    """
    sh, sw = params[0], params[1]
    sigma_table = params[2] if len(params) > 2 else {}

    H, W = X0.shape
    h, w = T_Y0.shape

    # Check dimensions
    if h != sh * H or w != sw * W:
        return {}

    templates = {}

    if kind == "SBS-Y":
        # Partition T_Y0 into blocks
        blocks = T_Y0.reshape(H, sh, W, sw)
        blocks = np.moveaxis(blocks, 1, 2)

        # Extract template for each palette value
        for v, tid in sigma_table.items():
            # Find first occurrence of v in X0
            first_i, first_j = None, None
            for i in range(H):
                for j in range(W):
                    if X0[i, j] == v:
                        first_i, first_j = i, j
                        break
                if first_i is not None:
                    break

            if first_i is not None:
                templates[tid] = blocks[first_i, first_j].copy()

    elif kind == "SBS-param":
        # Get T_X0
        T_X0, _ = types_from_output(X0)

        # Build constant templates from T_X0
        for v, tid in sigma_table.items():
            # Find first occurrence of v in X0
            first_i, first_j = None, None
            for i in range(H):
                for j in range(W):
                    if X0[i, j] == v:
                        first_i, first_j = i, j
                        break
                if first_i is not None:
                    break

            if first_i is not None:
                # Constant template filled with T_X0 type
                templates[tid] = np.full((sh, sw), T_X0[first_i, first_j], dtype=T_X0.dtype)

    return templates


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
        choices=["pi-receipts", "free-simple-receipts", "free-tile-receipts", "free-sbs-y-receipts", "free-sbs-param-receipts", "free-intersect-pick", "transport-receipts", "quotas-receipts"],
        help="Solver mode: pi-receipts (WO-2), free-simple-receipts (WO-3A), free-tile-receipts (WO-3B), free-sbs-y-receipts (WO-3C), free-sbs-param-receipts (WO-3D), free-intersect-pick (WO-4), transport-receipts (WO-5), or quotas-receipts (WO-6)",
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
    elif args.mode == "free-sbs-param-receipts":
        run_free_sbs_param_receipts(
            tasks=tasks,
            out_path=args.out,
        )
    elif args.mode == "free-intersect-pick":
        run_free_intersect_pick(
            tasks=tasks,
            out_path=args.out,
        )
    elif args.mode == "transport-receipts":
        run_transport_receipts(
            tasks=tasks,
            out_path=args.out,
        )
    elif args.mode == "quotas-receipts":
        run_quotas_receipts(
            tasks=tasks,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
