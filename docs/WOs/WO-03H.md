# WO-3H — FREE Verifier: Stripe / Column / Row Projections (types-only)

## Anchors to read first

* `00_math_spec.md` §5 **Shape Map (FREE transport)** — “known projection (e.g., 3×7→3×3)” and more generally, **fixed index maps** along rows/columns on the **type mosaic** (Π).
* `01_math_spec_addendum.md` — FREE proofs act **on types**; candidates are proved per training pair and **intersected** across pairs in WO-4; receipts must show exact equality (no heuristic scores).

This WO formalizes those “projection”/“band map” cases as **deterministic index maps** along a single axis, proven by exact equality on **type mosaics** (T_X = Π(X), T_Y = Π(Y)).

---

## Goal

For each training pair ((X \to Y)), **prove** that the output type mosaic (T_Y) is obtained from the input type mosaic (T_X) by applying a **single, fixed index map** along exactly one axis (rows or columns), uniformly for all lines on that axis. Examples the verifier must cover:

1. **Subset selection**: choose explicit indices on an axis (e.g., 3×7→3×3 left or right columns).
2. **Stride selection**: pick every (k)-th row/column with fixed offset.
3. **Interleave selection**: repeat a fixed pattern of positions each period across the axis.
   (Optionally) 4) **Fold**: treat as a specific subset selection (left/right block of fixed width).

**Everything is types-only**: you operate on (T_X, T_Y) from WO-1.

---

## Exact libraries & calls (no custom algorithms)

Use only mature, documented NumPy APIs:

* **Uniform extraction along an axis**: `numpy.take(arr, indices, axis=AXIS, mode='wrap')` (documented; `mode='wrap'` defines well-posed periodic selection) .
* **Exact equality** of the constructed projection to (T_Y): `numpy.array_equal(A, B)` (documented) .
* **Vector building**: `numpy.arange`, `numpy.tile`, `numpy.concatenate` (documented) for generating index sequences and periodic patterns .

No hand-rolled sliding windows, no tolerance comparisons, no custom “similarity” metrics.

---

## IO (module API)

Create a new module `arc/free_stripe.py`:

```python
from typing import List, Tuple, Optional, Union
import numpy as np

# pattern_spec is a small, immutable tuple describing the map:
#   ("subset", axis, tuple(indices))
#   ("stride", axis, stride_k, offset_o)
#   ("interleave", axis, period_p, tuple(select_positions))
#   ("fold", axis, side, block_len)         # optional; implemented as subset indices
Pattern = Tuple[Union[str, int], ...]
Candidate = Tuple[str, Pattern]            # ("band_map", pattern_spec)

def verify_stripe_maps(T_X: np.ndarray, T_Y: np.ndarray) -> List[Candidate]:
    """
    Return all proven band-map candidates ("band_map", pattern_spec) for this pair.
    Operates strictly on type mosaics. No color access.
    """
```

Return **all** matching candidates (usually 0 or 1). The runner never calls this module directly; **WO-4** consumes its outputs.

---

## Pattern families to check (precise)

All checks build a candidate `indices` vector and then verify:

```python
T_hat = np.take(T_X, indices, axis=AXIS, mode="wrap")  # documented behavior
ok = np.array_equal(T_hat, T_Y)
```

If `ok` is true, emit `("band_map", pattern_spec)`.

1. **Subset**:

   * AXIS ∈ {0 (rows), 1 (cols)}.
   * For every plausible slice family you want to cover:

     * **Left block** (`fold-left`): `indices = np.arange(dst_len)`
     * **Right block** (`fold-right`): `indices = src_len - dst_len + np.arange(dst_len)`
     * (Optional) any fixed provided index lists (e.g., for 3×7→3×3 pick `[0,1,2]` or `[4,5,6]`).
   * Verify with `np.take` + `array_equal`.

2. **Stride**:

   * For each AXIS and for strides (k) in a small set (e.g., 2..min(4, src_len)):

     * For offsets (o \in [0, k-1]), build `indices = np.arange(o, src_len, k)`; require `len(indices) == dst_len` else skip.
     * Verify via `np.take(indices, axis=AXIS, mode='wrap')` and `array_equal`.

3. **Interleave (periodic selection)**:

   * For each AXIS and small `period p` (e.g., 2..min(6, src_len)), try a few simple `select_positions` patterns within a period (e.g., `[0]`, `[0,1]`, `[0,2]`, etc.).
   * Build full index sequence by tiling:

     ```python
     base = np.array(select_positions) % p
     reps = int(np.ceil(dst_len / len(base)))
     indices = np.tile(base, reps)[:dst_len] + p * np.repeat(np.arange(dst_len // len(base) + 1), len(base))[:dst_len]
     indices %= src_len
     ```
   * Verify with `np.take(..., mode='wrap')` + `array_equal`.

> All of these maps are **“constant across lines”** because `indices` is reused unchanged for every line of the orthogonal axis — exactly the anchor’s “fixed pattern” requirement. Equality is checked on the whole array, so there is no risk of per-line drift.

**Notes:**

* Keep the search space **bounded** (e.g., small `k` and `p`) to avoid pathological runtimes; ARC grids are tiny, so O(H·W·candidates) is fine on CPU.
* Each candidate is an **exact** equality check; no heuristics.

---

## Receipts (first-class)

Per training pair, write one JSON object into the existing receipts stream:

```json
{
  "task_id": "…",
  "pair_index": 0,
  "free_stripe": {
    "source_shape": [H, W],
    "target_shape": [h, w],
    "candidates": [
      {
        "axis": 1,
        "mode": "subset",
        "indices": [0,1,2],
        "ok": true
      },
      {
        "axis": 1,
        "mode": "stride",
        "stride": 2,
        "offset": 0,
        "ok": false
      },
      {
        "axis": 0,
        "mode": "interleave",
        "period": 3,
        "select": [0,2],
        "ok": false
      }
    ]
  }
}
```

If at least one candidate has `ok=true`, additionally emit:

```json
"candidate": ["band_map", ["subset", 1, [0,1,2]]]
```

(Use the exact `pattern_spec` you’ll hand to WO-4.)

---

## WO-4 wiring (explicit)

**Extend WO-4** to include `verify_stripe_maps` in the per-pair candidate collection **after** `h/v-concat*` (rank 1) and **before** `tile`/`SBS*`:

* Add a new rank **between** rank-1 transforms and `tile`:

```
order rank:
  0: "identity"
  1: "h-mirror-concat", "v-double", "h-concat-dup", "v-concat-dup"
  2: "band_map"                 <-- NEW (this WO)
  3: "tile"
  4: "SBS-Y"
  5: "SBS-param"
```

* In the per-pair collector: `stripe_cands = verify_stripe_maps(T_X, T_Y)`; extend the candidate set with all returned `("band_map", pattern_spec)` tuples.
* The rest of WO-4 (intersection + pick) is unchanged.

This lets you **test impact immediately**: run WO-4 and watch how many previously `identity` tasks now prove a stronger `band_map` terminal, which should reduce the “POLICY” bucket downstream.

---

## Runner changes (minimal)

* Add a **receipts mode** to call only this verifier on all pairs:

```bash
python -m arc.cli \
  --mode free-stripe-receipts \
  --challenges /mnt/data/arc-agi_training_challenges.json \
  --out outputs/receipts_wo3h.jsonl
```

* Then run:

```bash
python -m arc.cli --mode free-intersect-pick \
  --challenges … --out outputs/receipts_wo4.jsonl

python -m arc.cli --mode v0 \
  --policy meet \
  --challenges … --pred … --receipts … --report … \
  --with-gt --gt /path/to/gt.json
```

No change to **WO-5/6/7** or the v0 runner logic.

---

## Reviewer instructions (to gauge structural impact and policy→match movement)

1. **Verify the verifier** (WO-3H receipts):

   * Ensure each pair logs `source/target_shape` and a small set of candidates.
   * Spot-check a few where `ok=true` by re-applying the same `indices` with `np.take(..., mode='wrap')` on the saved (T_X) and confirming `array_equal` is true.

2. **Check structural pick** (WO-4):

   * Compare **candidate union** and **intersected** lists **before vs after** adding WO-3H.
   * Count how many tasks switched from `identity` (rank 0) to `band_map` (rank 2).

3. **Run v0 end-to-end** and compare **triage**:

   * **MATCH** count — should increase if structure now fixes color degrees of freedom.
   * **MISMATCH/POLICY** — should **decrease** as band maps constrain K.
   * **MISMATCH/IMPLEMENTATION** stays 0 (receipts green), **FREE_DIFFERS** remains ~0 unless GT prefers a different terminal.
   * Report: `ΔMATCH`, `ΔPOLICY`, with example task ids where `identity → band_map` led to MATCH.

This is the exact progressive track to show **POLICY → MATCH** shift driven by a stronger FREE proof, which is what the anchors predict.

---

## Pass criteria

* Per-pair receipts for **all 1000×train-pairs** produced; candidates are small and exact (no false positives).
* WO-4 consumes `band_map` and logs it at **rank 2**.
* End-to-end v0 runs deterministically; summary shows **increase in `band_map` picks** and a **reduction in POLICY mismatches**, with at least some **new MATCH**.

---

## Anti-bloat & determinism

* The verifier is O(H·W·N_cand) with small, fixed `N_cand`; everything is vectorized with NumPy.
* No new business logic in the runner; only calls into `verify_stripe_maps` (WO-3H) and the existing WO-4/WO-7 plumbing.
* All receipts are **one JSON object per line** (JSON Lines) for easy audit.

---

