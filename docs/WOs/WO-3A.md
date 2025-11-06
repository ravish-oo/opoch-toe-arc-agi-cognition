# WO-3A — FREE Verifiers: Identity / Mirror-Concat / V-Double / Concat-Dup

## Anchors to read first

* `@docs/anchors/00_math_spec.md` — §5 *Shape Map (Free Transport)*: simple, content-level verifiable morphisms.
* `@docs/anchors/01_math_spec_addendum.md` — confirms these are FREE, verifiable from a single pair with no quotas.

## Goal

For **each training pair** ((X!\to Y)), deterministically verify a set of **simple FREE terminals** at **color level** (no types needed here):

* `identity`  — shape equality only (FREE “do nothing” shape),
* `h-mirror-concat` — either `[rev(X) | X]` or `[X | rev(X)]`,
* `v-double`  — vertical doubling (we define as `[X ; X]` for v0),
* `h-concat-dup` — horizontal duplicate `[X | X]`,
* `v-concat-dup` — vertical duplicate `[X ; X]`.

Return **all** candidates that match exactly for that pair (no heuristics). These receipts will feed WO-4 (intersection & pick).

---

## Interfaces (frozen)

```python
# arc/free_simple.py
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

Kind = str      # "identity", "h-mirror-concat", "v-double", "h-concat-dup", "v-concat-dup"
Params = Optional[Tuple[Any, ...]]  # e.g., ("rev|id") for h-mirror-concat variants

def verify_simple_free(X: np.ndarray, Y: np.ndarray) -> List[Tuple[Kind, Params]]:
    """Return all simple FREE candidates verified exactly for this (X->Y)."""
```

* Deterministic output ordering: return candidates in this order if they match:
  `identity`, `h-mirror-concat(rev|id)`, `h-mirror-concat(id|rev)`, `v-double`, `h-concat-dup`, `v-concat-dup`.

---

## Exact libraries & functions (no custom algorithms)

Use **NumPy** only; no loops except for the two mirror variants. All checks are vectorized equality or documented array ops:

* **Equality**: `numpy.array_equal(a, b)` (shape must match and every value equal) ([numpy.org][1]).
* **Left/right flip** (rev for rows): `numpy.fliplr(X)` (O(1) view; axis-1 reversed) ([numpy.org][2]).
* **Up/down flip** (not strictly needed here, but listed for completeness): `numpy.flipud(X)` (O(1) view; axis-0 reversed) ([numpy.org][3]).
* **Concatenation**: `numpy.concatenate([A, B], axis=1)` for horizontal; `axis=0` for vertical ([numpy.org][4]).
* General flipping catalogue (reference page): `numpy.flip`, `fliplr`, `flipud`, `rot90` live under array-manipulation routines ([numpy.org][5]).

> We deliberately **do not** use `allclose` or tolerances—ARC colors are exact integers; only `array_equal` qualifies here ([numpy.org][6]).

---

## Verification logic (precise)

Let ((H,W)=X.shape) and ((h,w)=Y.shape).

1. **identity**

   * *Rule*: shape equality proves the FREE terminal; content is handled by quotas later.
   * Check: `(h, w) == (H, W)` → add `("identity", None)`.

2. **h-mirror-concat**
   Two symmetric variants; both require `h == H and w == 2*W`:

   * `("rev|id")`: `array_equal(Y[:, :W], fliplr(X))` **and** `array_equal(Y[:, W:], X)` ([numpy.org][2]).
   * `("id|rev")`: `array_equal(Y[:, :W], X)` **and** `array_equal(Y[:, W:], fliplr(X))` ([numpy.org][2]).

3. **v-double** (v0 definition = duplicate X vertically)

   * Require `h == 2*H and w == W` **and**
     `array_equal(Y[:H, :], X)` **and** `array_equal(Y[H:, :], X)`.

4. **h-concat-dup**

   * Require `h == H and w == 2*W` **and**
     `array_equal(Y, concatenate([X, X], axis=1))` ([numpy.org][4]).

5. **v-concat-dup**

   * Require `h == 2*H and w == W` **and**
     `array_equal(Y, concatenate([X, X], axis=0))` ([numpy.org][4]).

> Note: `v-double` and `v-concat-dup` are **both** emitted if both definitions pass (content-duplicate vs. whole-array duplicate). WO-4’s frozen pick order will choose deterministically. No heuristics.

---

## Receipts (first-class)

For **each train pair**, write one JSON object in the task’s receipts block (WO-2 writer), e.g.:

```json
{
  "task_id": "6d0aefbc",
  "pair_index": 0,
  "free_simple": {
    "identity": {"shape_eq": true},
    "h_mirror_concat": {
      "rev|id": {"match": true},
      "id|rev": {"match": false}
    },
    "v_double": {"match": false},
    "h_concat_dup": {"match": true},
    "v_concat_dup": {"match": false}
  },
  "candidates": [
    ["identity", null],
    ["h-mirror-concat", "rev|id"],
    ["h-concat-dup", null]
  ]
}
```

* `candidates` must mirror exactly what `verify_simple_free` returns (order preserved).
* Every boolean is a direct NumPy equality or shape predicate; no computed “scores,” so the reviewer can audit the truth table against the rules.

**Task-level rollup (union only in WO-3A):**
In the final summary per task, also include a unioned set of candidate kinds (WO-4 will intersect across pairs, not WO-3A).

---

## Runner changes

Add a new mode to the runner:

```bash
python -m arc.solve \
  --mode free-simple-receipts \
  --challenges /mnt/data/arc-agi_training_challenges.json \
  --out outputs/receipts_wo3a.jsonl
```

Behavior:

* Load tasks (as in WO-2).
* For **each training pair**, call `verify_simple_free(X, Y)`.
* Append a JSON object (as above) per pair.
* After all pairs in a task, append a compact task-level union record:

  ```json
  {"task_id":"6d0aefbc","free_simple_union":["identity","h-mirror-concat","h-concat-dup"]}
  ```

CLI and JSONL conventions are the same as WO-2 (use `argparse`, `pathlib.Path`, `hashlib` for any needed hashes; one JSON object per line per JSONL spec) ([Python documentation][7]).

---

## Reviewer instructions (all 1000 tasks)

1. **Run**:

   ```bash
   python -m arc.solve \
     --mode free-simple-receipts \
     --challenges /mnt/data/arc-agi_training_challenges.json \
     --out outputs/receipts_wo3a.jsonl
   ```
2. **Audit**:

   * Confirm the file has **one record per train pair** plus one union record per task.
   * Spot-check a few tasks by recomputing the conditions from the spec (e.g., ensure that whenever `h==H && w==2W` and `Y[:, :W]==fliplr(X) && Y[:, W:]==X`, the `h-mirror-concat.rev|id` flag is `true`) using the receipts’ booleans alone—no code reading needed.
3. **Interpreting any “gaps”**:

   * There are **no spec gaps** in WO-3A—simple verifiers either match or they don’t.
   * If a candidate should be `true` by the rule but shows `false`, that is an **implementation bug** (likely misuse of `concatenate`, `fliplr`, or a shape predicate).
   * If none match for a pair, that’s normal; WO-3A is only a subset of the FREE library.

No heuristics, no shortcuts. If performance ever becomes an issue (unlikely—these are O(HW) checks), batching the equality tests is permissible, but only with the same NumPy calls.

---

## Acceptance criteria (green = WO-3A done)

* ✔ Runs on the full corpus without error; produces JSONL with per-pair and per-task records.
* ✔ Every candidate boolean is computed using **only**: `array_equal`, `fliplr`, `concatenate`, and shape comparisons.
* ✔ Deterministic order of `candidates`.
* ✔ No “approximate” matches or tolerances.
* ✔ Code `arc/free_simple.py` + small runner plumbing.

---

If this is good, I’ll proceed to WO-3B (types-periodic tile) with the same level of precision.

[1]: https://numpy.org/doc/2.1/reference/generated/numpy.array_equal.html?utm_source=chatgpt.com "numpy.array_equal — NumPy v2.1 Manual"
[2]: https://numpy.org/doc/2.2/reference/generated/numpy.fliplr.html?utm_source=chatgpt.com "numpy.fliplr — NumPy v2.2 Manual"
[3]: https://numpy.org/doc/2.2/reference/generated/numpy.flipud.html?utm_source=chatgpt.com "numpy.flipud — NumPy v2.2 Manual"
[4]: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html?utm_source=chatgpt.com "numpy.concatenate — NumPy v2.3 Manual"
[5]: https://numpy.org/doc/2.2/reference/routines.array-manipulation.html?utm_source=chatgpt.com "Array manipulation routines"
[6]: https://numpy.org/doc/1.15/reference/generated/numpy.array_equal.html?utm_source=chatgpt.com "numpy.array_equal — NumPy v1.15 Manual"
[7]: https://docs.python.org/3/library/argparse.html?utm_source=chatgpt.com "argparse — Parser for command-line options, arguments ..."
