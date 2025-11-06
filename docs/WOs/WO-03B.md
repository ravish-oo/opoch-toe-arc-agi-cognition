# WO-3B — FREE Verifier: Types-Periodic Tile

## Anchors to read first

* `docs/anchors/00_math_spec.md` — §5 *Shape Map (Free Transport)*: integer blow-ups as FREE morphisms proven from a single training pair (no quotas).
* `docs/anchors/01_math_spec_addendum.md` — affirms FREE verification must be done on **types** (Π(Y)), not on colors.

## Goal

Given a training pair ((X!\to Y)) and the **type mosaic** (T_Y=\Pi(Y)), prove a FREE **tile** morphism by checking that (Y) is an integer blow-up of a base (H\times W) pattern in **types**, i.e.

* (h \bmod H = 0), (w \bmod W = 0) with ((H,W)=X.shape), ((h,w)=Y.shape);
* for all ((r,c)), (T_Y[r,c] = T_Y[r \bmod H,; c \bmod W]).

Return exactly one candidate when true:

```python
("tile", (sh, sw))   # where sh = h//H, sw = w//W
```

No heuristics, no partial matches, no color-level checks.

---

## Interface (frozen)

```python
# arc/free_simple.py   (or arc/free_tile.py if you prefer split)
from typing import Optional, Tuple, Any
import numpy as np

def verify_tile_types(X: np.ndarray, Y: np.ndarray, T_Y: np.ndarray) -> Optional[Tuple[str, Tuple[int,int]]]:
    """
    Verify types-periodic tiling:
    - shapes: h%H==0 and w%W==0
    - types periodicity: T_Y == base tiled sh x sw
    Return ("tile",(sh,sw)) if proven, else None.
    """
```

Deterministic: return `None` or exactly one tuple. There is no param ambiguity for tiling.

---

## Exact libraries & functions to use (no custom algos)

Use **only** well-documented NumPy APIs:

* **Shape checks**: plain Python ints from `X.shape`, `Y.shape`.
* **Equality**: `numpy.array_equal(a, b)` — exact match, same shape, all elements equal. Do **not** use tolerances or `allclose`. ([numpy.org][1])
* **Construct the tiled expectation** (pick one of the two canonical vectorized routes; both are allowed):

  1. **np.tile** the (H\times W) base pattern to `(sh, sw)` and compare
     `np.tile(base, (sh, sw))` (repeat along both axes). ([numpy.org][2])
  2. **np.take with wrap** indices to compare without materializing the tile
     Build index vectors `rows = np.arange(h) % H`, `cols = np.arange(w) % W`; then
     `wrapped = T_Y.take(rows, axis=0, mode='wrap').take(cols, axis=1, mode='wrap')` and check `array_equal(T_Y, wrapped)`. ([numpy.org][3])
* Optional reference: **broadcasting** docs if you choose an advanced broadcast approach (not required). ([numpy.org][4])

Either approach is O(h·w) and pure CPU. Prefer (2) if you want to avoid allocating a full `np.tile` copy for very large grids; both are acceptable for ARC.

> Do **not** hand-roll loops; vectorize with the cited primitives only.

---

## Verification logic (precise, step-by-step)

1. Let `(H, W) = X.shape` and `(h, w) = Y.shape`.
   If `h % H != 0 or w % W != 0`: **return None** (not a multiple).

2. Compute `sh, sw = h // H, w // W`.

3. Extract the base types block (B := T_Y[0:H, 0:W])`.

4. Build the expected tiled types:

   * **Option A (tile)**: `T_expected = np.tile(B, (sh, sw))`. ([numpy.org][2])
   * **Option B (wrap-take)**:
     `rows = np.arange(h) % H; cols = np.arange(w) % W; T_expected = T_Y.take(rows, axis=0, mode='wrap').take(cols, axis=1, mode='wrap')`. ([numpy.org][3])

5. If `np.array_equal(T_Y, T_expected)`: **return `("tile",(sh, sw))`**, else **return None**. ([numpy.org][1])

No other conditions. We do not inspect colors, only types.

---

## Receipts (first-class)

For **each training pair** we evaluate, append a JSON object to the receipts stream:

```json
{
  "task_id": "025d127b",
  "pair_index": 0,
  "free_tile_types": {
    "H": 3, "W": 3, "h": 9, "w": 9,
    "sh": 3, "sw": 3,
    "periodic_by_types": true,          // result of array_equal(T_Y, T_expected)
    "method": "wrap-take"               // "tile" or "wrap-take", for audit
  },
  "candidate": ["tile", [3, 3]]
}
```

If the proof fails (not periodic), still emit the object with `periodic_by_types: false` and **no** `candidate` field. This makes audits uniform.

---

## Runner changes

Add a receipts mode to the CLI:

```bash
python -m arc.solve \
  --mode free-tile-receipts \
  --challenges /mnt/data/arc-agi_training_challenges.json \
  --out outputs/receipts_wo3b.jsonl
```

Behavior:

* For each task and each training pair:

  1. Compute `T_Y = arc.pi.types_from_output(Y)[0]`.
  2. Call `verify_tile_types(X, Y, T_Y)`.
  3. Emit the per-pair receipt shown above.

Optionally, after all pairs in a task, append a **task-level union** record of candidates (as in WO-3A). File format is JSON Lines — *one JSON object per line*; UTF-8; newline-delimited. ([JSON Lines][5])

---

## Reviewer instructions (all 1000 tasks)

1. **Run**:

   ```bash
   python -m arc.solve \
     --mode free-tile-receipts \
     --challenges /mnt/data/arc-agi_training_challenges.json \
     --out outputs/receipts_wo3b.jsonl
   ```
2. **Check**:

   * Each train pair record shows `(H,W,h,w)` and computed `(sh,sw)` whenever multiples.
   * If `periodic_by_types == true`, a `candidate: ["tile",[sh,sw]]` exists and `(h,w)==(sh*H, sw*W)` holds trivially.
   * Spot-check a few by re-computing `rows%H` / `cols%W` logic mentally — the receipt gives all numbers needed.
3. **Identify a legit gap vs. bug**:

   * There are **no spec gaps** here: either types are periodic or not.
   * False positives are unacceptable; this is an **exact** equality check (`array_equal`).
   * If periodicity should be true but is false, the fault is implementation (wrong slice for `B`, wrong indices/order, or comparing colors instead of types).

---

## Performance guidance (CPU)

* Both `np.tile` and `np.take(..., mode='wrap')` are vectorized and run in C. Broadcasting is not required; don’t invent it. If you do use broadcasting techniques, adhere to NumPy’s broadcasting rules. ([numpy.org][4])
* ARC grids are small; no premature optimization, no memory micro-tuning.

---

## Acceptance criteria (green = WO-3B done)

* ✔ Runner completes on the full corpus; JSONL has one record per training pair (and an optional task union record).
* ✔ Every **true** tile case emits exactly one `("tile",(sh,sw))` candidate; **no false positives**.
* ✔ Deterministic results across runs; method label recorded.
* ✔ Code ≤ ~120 LOC for the verifier + small runner plumbing; only uses the cited NumPy calls.

---

[1]: https://numpy.org/doc/2.1/reference/generated/numpy.array_equal.html?utm_source=chatgpt.com "numpy.array_equal — NumPy v2.1 Manual"
[2]: https://numpy.org/devdocs/reference/generated/numpy.tile.html?utm_source=chatgpt.com "numpy.tile — NumPy v2.4.dev0 Manual"
[3]: https://numpy.org/devdocs/reference/generated/numpy.take.html?utm_source=chatgpt.com "numpy.take — NumPy v2.4.dev0 Manual"
[4]: https://numpy.org/doc/stable/user/basics.broadcasting.html?utm_source=chatgpt.com "Broadcasting — NumPy v2.3 Manual"
[5]: https://jsonlines.org/?utm_source=chatgpt.com "JSON Lines"
