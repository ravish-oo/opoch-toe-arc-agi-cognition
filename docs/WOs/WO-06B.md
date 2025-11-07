# WO-6b — Multi-Y Quotas Meet (strictly types-aligned, semantic meet)

## Anchors to read first

* ` @docs/anchors/00_math_spec.md ` §4 **Per-Type Quotas (Paid Ledger)** and §6 **Least Fill (meet)**: across multiple trainings, **align types by φ** (equal φ ⇒ same type) and, if counts differ, take the **component-wise minimum** to remain admissible for **all** trainings.
* ` @docs/anchors/01_math_spec_addendum.md `: receipts; FREE vs PAID separation; no quotas from inputs.

We are implementing **exactly** that meet: (K^*(\phi,c) = \min_i K_i(\phi,c)).

---

## Goal

Compute a **global meet quotas table** (K^*) over **φ-type classes** (not over per-grid type ids), then adapt it to the test fill by mapping each **new** child type (S') (from WO-5) back to its **parent** Y₀ type (S), then to the parent’s **φ signature**, and finally to (K^*(\phi,\cdot)). No quotas are ever read from inputs.

---

## Public API (frozen)

```python
# arc/quotas_meet.py
from typing import Dict, Tuple, Any
import numpy as np

PhiKey = Tuple[int, ...]     # the φ feature tuple from WO-1 (center, 4N, 4x2-step, edge flags, parity)

def build_phi_key_index(Y: np.ndarray, T: np.ndarray) -> Dict[int, PhiKey]:
    """
    Map each parent type-id S in Π(Y) to its canonical φ key (equal φ ⇒ same type class).
    """

def quotas_per_phi_on_grid(Y: np.ndarray, T: np.ndarray, C: int) -> Dict[PhiKey, np.ndarray]:
    """
    For one training output Y with its types T, return per-φ color histograms (length-C vectors).
    """

def quotas_meet_all(train_pairs: list[dict], C: int) -> Dict[PhiKey, np.ndarray]:
    """
    For all training outputs in a task, align by φ and compute the component-wise minimum across trainings:
      K*(φ, :) = min_i  K_i(φ, :)
    Missing φ in a grid acts like a zero vector (admissibility for all).
    """

def quotas_for_test_parent(
    parent_of: Dict[int, int],                 # S' -> S, from WO-5
    parent_phi_map: Dict[int, PhiKey],         # S -> φ, from build_phi_key_index on Y0
    K_star_phi: Dict[PhiKey, np.ndarray],      # K*(φ,:)
) -> Dict[int, np.ndarray]:
    """
    Produce the final quotas dict for fill: map each child type S' to K*(φ(parent(S')), :).
    """
```

---

## Exact libraries & functions to use (no custom algos)

* **Unique by φ** (per type class): for any array, `numpy.unique` gives sorted unique values; with `return_counts=True` it provides counts (we’ll use that for palettes or verification as needed) ([NumPy][1]).
* **Per-type color histograms:** `numpy.bincount(colors, minlength=C)` to get a length-C vector for each type; it’s the canonical, documented histogram primitive for non-negative ints (ARC colors 0..9) ([NumPy][2]).
* **Meet across trainings:** stack same-φ vectors and take the **element-wise minimum**; use `numpy.minimum` or `numpy.ufunc.reduce` (`np.minimum.reduce([...], axis=0)`) ([NumPy][3]).
* **Join/stack** as needed: `numpy.stack` / `numpy.concatenate` for assembling arrays (documented) ([NumPy][4]).
* **Receipts hashing:** `hashlib.sha256(...).hexdigest()` (official docs) ([Python documentation][5]).
* **JSON Lines**: one JSON object **per line**, UTF-8 (official spec) ([Jsonlines][6]).
* **CLI plumbing**: stdlib `argparse` for mode flagging, `pathlib.Path` for I/O (official docs) ([Python documentation][7]).

No pandas, no custom grouping, no hand-rolled reductions.

---

## Implementation details (precise)

### 1) `build_phi_key_index(Y, T)`

* Use WO-1’s φ function to compute the **canonical φ tuple** for **one representative pixel** of each type (S). Get any index for that type:

  ```python
  # pick the first pixel of S
  idx = np.flatnonzero((T.ravel(order='C') == S))
  r, c = divmod(int(idx[0]), T.shape[1])
  key = phi_features(Y, r, c)      # from WO-1
  ```

  `np.flatnonzero` is standard and avoids Python loops.
  (You already have `phi_features` in WO-1; reuse it.)

### 2) `quotas_per_phi_on_grid(Y, T, C)`

* For each type (S) in `np.unique(T)`, build `colors = Y[T==S]` and compute `np.bincount(colors, minlength=C)`.
* Map the **φ key** from `build_phi_key_index` to that length-C vector. If multiple S share the same φ key (they shouldn’t; φ defines the type), sum their vectors.

### 3) `quotas_meet_all(train_pairs, C)`

* For each training output (Y_i): compute (T_i=\Pi(Y_i)) via WO-1, then call `quotas_per_phi_on_grid`.
* Build the **union** of φ keys across all trainings.
* For each φ:

  * Retrieve that grid’s vector if present; otherwise use a zero vector `np.zeros(C, dtype=np.int64)`.
  * Stack the vectors → an array shape `(num_present_or_zero, C)`.
  * Compute `K_star_phi[φ] = np.minimum.reduce(stack, axis=0)` (documented reduce) ([NumPy][8]).
* (Optional) Compress φ keys to a short **hash** for receipts (keep the tuple for internal use).

### 4) `quotas_for_test_parent(parent_of, parent_phi_map, K_star_phi)`

* For every child S′, get its parent S with `parent_of[S′]`, then `φ = parent_phi_map[S]`.
* Assign `K_final[S′] = K_star_phi.get(φ, zeros(C))`.
  (If φ never occurs in any training output, the meet is zero vector; that’s admissible and consistent with anchors.)

> This keeps WO-7 unchanged: it continues to receive a dict `S' -> length-C vector` for the fill.

---

## Receipts (first-class)

Emit **one quotas-meet receipt per task**:

```json
{
  "task_id": "…",
  "quotas_meet": {
    "C": 10,
    "num_trainings": m,
    "phi_keys_total": U,                // |⋃_i φ_i|
    "phi_from_y0": P,                   // |φ keys present in Y0|
    "samples": [
      {
        "phi_hash": "…",                // sha256 of serialized φ tuple (optional)
        "train_vectors": [[…],[…],…],   // K_i(φ,:) in training order (zeros if absent)
        "meet_vector": […],             // K*(φ,:)
        "sum_meet":  nn
      }
    ],
    "adapted_for_test": {
      "num_children": K,                // number of S'
      "size_match_pass": true           // ∀ S': |child| == sum(K_final[S'])
    }
  }
}
```

* `size_match_pass` is checked using WO-5’s `parent_of` + the child sizes in `T_test`.
* Also include a **policy echo**: `"policy":"multi-Y meet"` in case we need to distinguish from v0 single-Y selection downstream.

Hash any long vectors (as needed) with `hashlib.sha256` ([Python documentation][5]).

---

## Runner changes (minimal, NO extra logic)

* Add a new mode (or a flag for v0) to enable **meet quotas**:

  * `--policy meet` uses **WO-6b** (`quotas_meet_all` + `build_phi_key_index` + `quotas_for_test_parent`)
  * `--policy single` keeps WO-6 (current v0 policy).

**In `v0` mode, with `--policy meet`:**

1. Compute `K_star_phi = quotas_meet_all(task["train"], C=10)`.
2. Compute `parent_phi_map = build_phi_key_index(Y0, T_Y0)` (we still need Y0’s φ to adapt each child to φ).
3. Compute `K_final = quotas_for_test_parent(parent_of, parent_phi_map, K_star_phi)`.
4. Pass `K_final` to **WO-7**.
5. Append the quotas-meet receipt to the JSONL.

This keeps the runner tiny and uses only the tested functions.

---

## Reviewer instructions (full corpus)

Run with meet policy:

```bash
python -m arc.cli \
  --mode v0 \
  --policy meet \
  --challenges /mnt/data/arc-agi_training_challenges.json \
  --pred outputs/predictions.json \
  --receipts outputs/receipts_all.jsonl \
  --report outputs/report.jsonl \
  --with-gt --gt /mnt/data/arc-agi_training_solutions.json
```

Report **exactly**:

1. **MATCH count** (should increase materially over v0 single-Y).
2. For **MISMATCH**: split by `IMPLEMENTATION | POLICY | FREE_DIFFERS` using the same triage from WO-8. Expect **POLICY** to drop sharply; `FREE_DIFFERS` remains 0 for identity-dominant cases unless GT needs a v1/v2 FREE family.
3. **SKIPPED**: unchanged (FREE_UNPROVEN backlog).

Additionally confirm **receipts**:

* `size_match_pass == true` for all filled children.
* A few φ samples showing `meet_vector = np.minimum.reduce(train_vectors)` (the numbers visibly meet).
* Determinism: two runs produce identical receipts and predictions SHA256.

---

## Acceptance criteria (green = WO-6b done)

* Code ≤ ~150 LOC in `arc/quotas_meet.py` (plus ~10–20 LOC glue in runner for `--policy meet`).
* Uses **only**: `np.unique`, `np.bincount`, `np.minimum.reduce`, `np.stack/concatenate`, `hashlib.sha256`, JSON Lines, `argparse`, `pathlib`.
* Receipts prove φ-alignment, meet vectors, and test adaptation (`size_match_pass`).
* Runner remains thin (no new verifiers, no recomputation of WOs).
* Deterministic across runs.

---

[1]: https://numpy.org/doc/stable/reference/generated/numpy.unique.html?utm_source=chatgpt.com "numpy.unique — NumPy v2.3 Manual"
[2]: https://numpy.org/doc/2.1/reference/generated/numpy.bincount.html?utm_source=chatgpt.com "numpy.bincount — NumPy v2.1 Manual"
[3]: https://numpy.org/devdocs/reference/generated/numpy.minimum.html?utm_source=chatgpt.com "numpy.minimum — NumPy v2.4.dev0 Manual"
[4]: https://numpy.org/devdocs/reference/generated/numpy.stack.html?utm_source=chatgpt.com "numpy.stack — NumPy v2.4.dev0 Manual"
[5]: https://docs.python.org/3/library/hashlib.html?utm_source=chatgpt.com "hashlib — Secure hashes and message digests"
[6]: https://jsonlines.org/?utm_source=chatgpt.com "JSON Lines"
[7]: https://docs.python.org/3/library/argparse.html?utm_source=chatgpt.com "argparse — Parser for command-line options, arguments ..."
[8]: https://numpy.org/devdocs/reference/generated/numpy.ufunc.reduce.html?utm_source=chatgpt.com "numpy.ufunc.reduce — NumPy v2.4.dev0 Manual"
