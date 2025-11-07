"""
Harness + Receipts Runner (WO-2, WO-3A, WO-3B, WO-3C, WO-3D, WO-4, WO-5, WO-6, WO-7)

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
  - fill-receipts: Fill by Rank + Idempotence (WO-7)

CLI:
  python -m arc.solve --mode pi-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-simple-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-tile-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-sbs-y-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-sbs-param-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode free-intersect-pick --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode transport-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode quotas-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
  python -m arc.solve --mode fill-receipts --challenges path/to/tasks.json --out outputs/receipts.jsonl
"""

import argparse
import hashlib
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
from arc.quotas import generate_quotas_receipt, choose_Y0, quotas
from arc.quotas_meet import (
    build_phi_key_index,
    quotas_meet_all,
    quotas_for_test_parent,
    generate_quotas_meet_receipt,
)
from arc.fill import fill_by_rank, generate_fill_receipt, idempotence_check


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
            T_test_before, _ = _transport_h_mirror_concat(T_Y0, params)
        elif kind == "v-double":
            T_test_before, _ = _transport_v_double(T_Y0)
        elif kind == "h-concat-dup":
            T_test_before, _ = _transport_h_concat_dup(T_Y0)
        elif kind == "v-concat-dup":
            T_test_before, _ = _transport_v_concat_dup(T_Y0)
        elif kind == "tile":
            T_test_before, _ = _transport_tile(T_Y0, params)
        elif kind in ["SBS-Y", "SBS-param"]:
            T_test_before, _ = _transport_sbs(T_Y0, free_tuple[1], X_test, kind)
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


def run_fill_receipts(
    tasks: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Run WO-7 fill-receipts mode.

    For each FREE_PROVEN task:
      - Load T_test and parent_of from WO-5 (re-compute transport)
      - Load K from WO-6 receipts
      - Call fill_by_rank to get Y*
      - Generate receipt with block verification and idempotence check
      - Emit one JSONL line per task

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

    # Load WO-6 receipts to get quotas
    wo6_path = Path("outputs/receipts_wo6_train.jsonl")
    if not wo6_path.exists():
        logging.error(f"WO-6 receipts not found: {wo6_path}")
        raise SystemExit(1)

    wo6_receipts = {}
    with open(wo6_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            wo6_receipts[rec["task_id"]] = rec

    # Counters for summary
    total_tasks = len(tasks)
    processed_count = 0
    skipped_unproven = 0
    all_blocks_satisfied = 0
    all_idempotent = 0
    all_blocks_size_match = 0
    all_no_oob = 0

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

        # Get WO-6 quotas
        if task_id not in wo6_receipts:
            skipped_unproven += 1
            continue

        wo6_rec = wo6_receipts[task_id]
        K_serialized = wo6_rec["quotas"]["K"]
        C = wo6_rec["quotas"]["C"]
        y0_index = wo6_rec["quotas"]["y0_index"]

        # Convert K from serialized format
        K = {int(type_id): np.array(counts, dtype=np.int64)
             for type_id, counts in K_serialized.items()}

        # Get chosen terminal
        chosen = free_inter["chosen"]
        kind, params = chosen[0], chosen[1]

        # Get training pairs
        train_pairs = task_data.get("train", [])
        if len(train_pairs) == 0:
            continue

        # Use Y₀ selected by WO-6 (not hardcoded [0]!)
        if y0_index >= len(train_pairs):
            continue

        X0 = np.array(train_pairs[y0_index]["input"], dtype=np.int32)
        Y0 = np.array(train_pairs[y0_index]["output"], dtype=np.int32)
        T_Y0, _ = types_from_output(Y0)

        # Get test input
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
            sh, sw = params[0], params[1]
            sigma_table = params[2] if len(params) > 2 else {}
            free_tuple = (kind, (sh, sw, sigma_table, templates))
        else:
            free_tuple = (kind, params)

        # Transport types to get T_test and parent_of
        T_test, parent_of = transport_types(T_Y0, free_tuple, X_test.shape, X_test)

        # Fill by rank
        Y_star = fill_by_rank(T_test, parent_of, K, C)

        # Generate receipt
        receipt = generate_fill_receipt(task_id, T_test, parent_of, K, Y_star, C)

        # Track statistics
        blocks_data = receipt["fill"]["blocks"]
        all_satisfied = all(b["quota_satisfied"] for b in blocks_data)
        if all_satisfied:
            all_blocks_satisfied += 1

        if receipt["fill"]["idempotent"]:
            all_idempotent += 1

        # Track conservation law violations
        all_size_match = all(b.get("size_match", True) for b in blocks_data)
        no_oob_colors = all(not b.get("oob_color", False) for b in blocks_data)
        if all_size_match:
            all_blocks_size_match += 1
        if no_oob_colors:
            all_no_oob += 1

        receipts.append(receipt)
        processed_count += 1

    # Write receipts
    write_jsonl(out_path, receipts)

    # Print summary
    logging.info("=" * 60)
    logging.info("WO-7 Fill Receipts Summary")
    logging.info("=" * 60)
    logging.info(f"Total tasks: {total_tasks}")
    logging.info(f"Processed (FREE_PROVEN): {processed_count}")
    logging.info(f"Skipped (unproven): {skipped_unproven}")
    logging.info(f"All blocks quota_satisfied: {all_blocks_satisfied}/{processed_count}")
    logging.info(f"All blocks size_match (conservation law): {all_blocks_size_match}/{processed_count}")
    logging.info(f"All no OOB colors: {all_no_oob}/{processed_count}")
    logging.info(f"All idempotent: {all_idempotent}/{processed_count}")
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


def _collect_per_pair_candidates(task: Dict[str, Any]) -> Tuple:
    """Helper to collect per-pair candidates for prove_free (reuses WO-3 logic)."""
    train_pairs = task.get("train", [])
    per_pair_simple, per_pair_tile, per_pair_sbs_y, per_pair_sbs_p = [], [], [], []

    for pair in train_pairs:
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

    return per_pair_simple, per_pair_tile, per_pair_sbs_y, per_pair_sbs_p


def run_v0(tasks: Dict[str, Dict[str, Any]], pred_path: Path, receipts_path: Path,
           report_path: Path, policy: str = "single", with_gt: bool = False, gt_path: Path = None) -> None:
    """
    WO-8: End-to-end v0 runner (thin orchestration only).

    Chains WO-4 → WO-6/WO-6b → WO-5 → WO-7 without re-implementing any WO logic.

    Args:
        policy: "single" (WO-6, single Y0) or "meet" (WO-6b, multi-Y meet)
    """
    predictions = []
    receipts = []
    terminal_counts = {}

    if with_gt and gt_path:
        # GT solutions file is {task_id: [[grid]]} format
        with open(gt_path, "r") as f:
            gt_tasks = json.load(f)
    else:
        gt_tasks = {}

    for task_id, task in tasks.items():
        # WO-4: prove FREE (collect candidates using WO-3 verifiers)
        candidates = _collect_per_pair_candidates(task)
        status4, payload4 = prove_free(task, *candidates)
        if status4 == "FREE_UNPROVEN":
            receipts.append({"task_id": task_id, "triage": {"status": "SKIPPED", "reason": "FREE_UNPROVEN"}})
            continue

        chosen = payload4["chosen"]
        free_tuple = chosen  # Already in (kind, params) format from prove_free
        terminal_counts[chosen[0]] = terminal_counts.get(chosen[0], 0) + 1

        # WO-6 / WO-6b: quotas (policy-dependent)
        y0_index = choose_Y0(task)
        Y0 = np.array(task["train"][y0_index]["output"], dtype=np.int32)
        T_Y0, _ = types_from_output(Y0)

        # WO-5: transport types (need this before quotas for meet policy)
        X_test = np.array(task["test"][0]["input"], dtype=np.int32)
        T_test, parent_of = transport_types(T_Y0, free_tuple, X_test.shape, X_test)

        # Compute quotas based on policy
        if policy == "meet":
            # WO-6b: Multi-Y meet quotas with admissibility check and fallback
            K_star_phi = quotas_meet_all(task["train"], C=10)
            parent_phi_map = build_phi_key_index(Y0, T_Y0)

            # Compute Y0 quotas (fallback when meet inadmissible)
            Y0_quotas = quotas(Y0, T_Y0, C=10)

            # Compute parent block sizes (for admissibility check: sum(K*) == |S|)
            parent_sizes = {}
            for parent_id in np.unique(T_Y0).tolist():
                parent_sizes[parent_id] = int(np.sum(T_Y0 == parent_id))

            # Adapt meet quotas to test with admissibility check
            K, phi_metadata = quotas_for_test_parent(
                parent_of, parent_phi_map, K_star_phi, parent_sizes, Y0_quotas
            )
        else:
            # WO-6: Single Y₀ quotas
            K = quotas(Y0, T_Y0, C=10)
            phi_metadata = {}  # Not used in single-Y policy

        # WO-7: fill + idempotence
        Y_star = fill_by_rank(T_test, parent_of, K, C=10)
        idempotent = idempotence_check(Y_star, C=10)

        # Generate WO-6/WO-6b and WO-7 receipts to check for IMPLEMENTATION failures
        if policy == "meet":
            quotas_receipt = generate_quotas_meet_receipt(
                task_id, task, K_star_phi, parent_phi_map, K, T_test, parent_of, phi_metadata, C=10
            )
        else:
            quotas_receipt = generate_quotas_receipt(task, task_id)
        fill_receipt = generate_fill_receipt(task_id, T_test, parent_of, K, Y_star, C=10)

        # Check receipts for failures
        fill_blocks = fill_receipt["fill"]["blocks"]
        all_quota_satisfied = all(b["quota_satisfied"] for b in fill_blocks)
        all_size_match = all(b.get("size_match", True) for b in fill_blocks)
        no_oob_colors = all(not b.get("oob_color", False) for b in fill_blocks)
        receipts_green = all_quota_satisfied and all_size_match and no_oob_colors and idempotent

        # SHA256 of prediction
        sha256_pred = hashlib.sha256(Y_star.astype(np.int64).tobytes()).hexdigest()

        # Write prediction
        predictions.append({"id": task_id, "output": Y_star.tolist(), "sha256": sha256_pred})

        # Build full receipt with detailed WO-6/WO-6b and WO-7 receipts
        receipt = {
            "task_id": task_id,
            "quotas": quotas_receipt.get("quotas") if policy == "single" else quotas_receipt.get("quotas_meet"),
            "fill": fill_receipt["fill"]
        }

        # Triage (if GT available)
        if with_gt and task_id in gt_tasks:
            # GT format is [[grid]] - extract grid
            GT = np.array(gt_tasks[task_id][0], dtype=np.int32)
            sha256_gt = hashlib.sha256(GT.astype(np.int64).tobytes()).hexdigest()

            if np.array_equal(Y_star, GT):
                triage = {"status": "MATCH", "reason": None}
            else:
                # Diagnose GT using prove_free (reuses WO-4 logic)
                synth_task = {"train": [{"input": X_test.tolist(), "output": GT.tolist()}], "test": [{"input": X_test.tolist()}]}
                gt_candidates = _collect_per_pair_candidates(synth_task)
                gt_status, gt_payload = prove_free(synth_task, *gt_candidates)
                gt_free = gt_payload["chosen"] if gt_status == "FREE_PROVEN" else None

                if gt_free is None or gt_free != chosen:
                    triage = {"status": "MISMATCH", "reason": "FREE_DIFFERS", "gt_free": gt_free}
                elif not receipts_green:
                    # IMPLEMENTATION: any receipts from WO-5/6/7 were not green
                    triage = {"status": "MISMATCH", "reason": "IMPLEMENTATION"}
                else:
                    triage = {"status": "MISMATCH", "reason": "POLICY"}

            triage.update({"free_chosen": chosen, "gt_free": gt_free if 'gt_free' in locals() else chosen,
                           "sha256_prediction": sha256_pred, "sha256_gt": sha256_gt})
            receipt["triage"] = triage

        receipts.append(receipt)

    # Write outputs
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    write_jsonl(receipts_path, receipts)

    # Report
    counts = {"MATCH": 0, "MISMATCH": 0, "SKIPPED": 0}
    mismatch_breakdown = {"IMPLEMENTATION": 0, "POLICY": 0, "FREE_DIFFERS": 0}
    for r in receipts:
        if "triage" not in r:
            continue  # Skip receipts without triage (no GT)
        status = r["triage"]["status"]
        counts[status] = counts.get(status, 0) + 1
        if status == "MISMATCH":
            reason = r["triage"].get("reason", "UNKNOWN")
            mismatch_breakdown[reason] = mismatch_breakdown.get(reason, 0) + 1

    report = [
        {"metric": "counts_by_status", **counts},
        {"metric": "mismatch_breakdown", **mismatch_breakdown},
        {"metric": "terminal_distribution", **terminal_counts},
        {"metric": "frozen_order_echo", "order": ["identity", ["h-mirror-concat", "v-double", "h-concat-dup", "v-concat-dup"], "tile", "SBS-Y", "SBS-param"]}
    ]
    write_jsonl(report_path, report)

    logging.info(f"v0 completed: {len(predictions)} predictions, {counts['MATCH']} MATCH, {counts['MISMATCH']} MISMATCH, {counts['SKIPPED']} SKIPPED")


def run_audit(task_id: str, receipts_path: Path) -> None:
    """WO-8: Audit mode - print receipt for given task ID."""
    with open(receipts_path, "r") as f:
        for line in f:
            receipt = json.loads(line)
            if receipt.get("task_id") == task_id:
                print(json.dumps(receipt, indent=2))
                return
    logging.error(f"Task {task_id} not found in receipts")


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
        choices=["pi-receipts", "free-simple-receipts", "free-tile-receipts", "free-sbs-y-receipts", "free-sbs-param-receipts", "free-intersect-pick", "transport-receipts", "quotas-receipts", "fill-receipts", "v0", "audit"],
        help="Solver mode: pi-receipts (WO-2), free-simple-receipts (WO-3A), free-tile-receipts (WO-3B), free-sbs-y-receipts (WO-3C), free-sbs-param-receipts (WO-3D), free-intersect-pick (WO-4), transport-receipts (WO-5), quotas-receipts (WO-6), fill-receipts (WO-7), v0 (WO-8 end-to-end), or audit (WO-8 receipt lookup)",
    )

    parser.add_argument(
        "--challenges",
        type=Path,
        required=False,
        help="Path to ARC challenges JSON file (not required for audit mode)",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        help="Output path for receipts JSONL file (for WO-2 through WO-7 modes)",
    )

    parser.add_argument(
        "--include-test-pi",
        action="store_true",
        help="Include test input grids in Π receipts (for sanity checking)",
    )

    # WO-8 v0 mode arguments
    parser.add_argument(
        "--pred",
        type=Path,
        help="Output path for predictions JSON (v0 mode)",
    )

    parser.add_argument(
        "--receipts",
        type=Path,
        help="Output path for receipts JSONL (v0 mode / audit mode input)",
    )

    parser.add_argument(
        "--report",
        type=Path,
        help="Output path for report JSONL (v0 mode)",
    )

    parser.add_argument(
        "--with-gt",
        action="store_true",
        help="Enable ground truth triage (v0 mode)",
    )

    parser.add_argument(
        "--gt",
        type=Path,
        help="Path to ground truth solutions JSON (v0 mode)",
    )

    # WO-6b policy argument
    parser.add_argument(
        "--policy",
        type=str,
        choices=["single", "meet"],
        default="single",
        help="Quotas policy: 'single' (WO-6, single Y0) or 'meet' (WO-6b, multi-Y meet)",
    )

    # WO-8 audit mode arguments
    parser.add_argument(
        "--id",
        type=str,
        help="Task ID to audit (audit mode)",
    )

    args = parser.parse_args()

    # Load tasks (not needed for audit mode)
    if args.mode == "audit":
        tasks = None
    else:
        if not args.challenges or not args.challenges.exists():
            logging.error(f"Challenges file not found or not specified: {args.challenges}")
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
    elif args.mode == "fill-receipts":
        run_fill_receipts(
            tasks=tasks,
            out_path=args.out,
        )

    elif args.mode == "v0":
        run_v0(
            tasks=tasks,
            pred_path=args.pred,
            receipts_path=args.receipts,
            report_path=args.report,
            policy=args.policy,
            with_gt=args.with_gt,
            gt_path=args.gt,
        )

    elif args.mode == "audit":
        run_audit(
            task_id=args.id,
            receipts_path=args.receipts,
        )


if __name__ == "__main__":
    main()
