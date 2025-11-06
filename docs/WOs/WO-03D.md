# WO-3D — FREE Verifier: **SBS-Param** (templates from Π(X))

## Anchors to read first

* ` @docs/anchors/00_math_spec.md ` — §5 *Shape Map (Free Transport)*: FREE maps are verified from training pairs, proofs operate on **types** (Π), not colors.
* ` @docs/anchors/01_math_spec_addendum.md ` — SBS addendum and the parametric variant: block grid (sh, sw), template chosen only by input cell via finite selector σ; **templates here come from Π(X)** (e.g., the 007bbfb7 class).

## Goal

For a training pair ((X!\to Y)), with (X\in\mathcal C^{H\times W}) and (Y\in\mathcal C^{h\times w}), **prove SBS-Param** (FREE) using **input types**:

1. Compute (T_X=\Pi(X)) and (T_Y=\Pi(Y)).
2. Require (h=s_h H) and (w=s_w W) for integers (s_h, s_w).
3. Partition (T_Y) into an ((H \times W)) grid of ((s_h\times s_w)) blocks.
4. Build a finite selector (\sigma:\mathcal C\to{0,\dots,M-1}) such that for each (v) appearing in (X), **every** block of (T_Y) at positions ((i,j)) with (X_{ij}=v) is **exactly equal** (type-wise) to the ((s_h\times s_w)) **template cut** taken from the corresponding region in **(T_X)**:
   [
   \boxed{,T_Y[i s_h:(i+1)s_h,\ j s_w:(j+1)s_w];=;\text{Template}^{(\sigma(v))};=;T_X[\hat r:\hat r+s_h,\ \hat c:\hat c+s_w],}
   ]
   where the “template from Π(X)” is defined canonically below (no ambiguity).
5. If all equalities hold, return one candidate:

```python
("SBS-param", (sh, sw, sigma_table, template_hashes_from_PiX))
```

Strict: no heuristics, no colors, no quotas from inputs. **Types only.**

---

## Interface (frozen)

```python
# arc/free_sbs.py
from typing import Optional, Tuple, Dict, Any
import numpy as np

def verify_SBS_param(X: np.ndarray, Y: np.ndarray) \
  -> Optional[Tuple[str, Tuple[int, int, Dict[int, int], Dict[int, str]]]]:
    """
    Prove SBS-Param on types (templates from Π(X)). Return:
      ("SBS-param", (sh, sw, sigma_table, template_hashes))
    or None if not proven.
    """
```

* Deterministic: return **None** or exactly one tuple.
* This verifier computes Π for X and Y inside (via WO-1); it uses **types only**.

---

## Exact libraries & functions to use (no custom algorithms)

Use mature, well-documented NumPy/stdlib APIs:

* **Π typing**: call WO-1 `types_from_output` to get `T_X`, `T_Y`.
* **Equality**: `numpy.array_equal(a, b)` for exact match (no tolerance) ([NumPy][1]).
* **Divisibility/shape**: integers from `.shape`.
* **Block reshaping** of `T_Y`:

  * Reshape + move axes (fast, clear):

    ```python
    blocks = T_Y.reshape(H, sh, W, sw)       # returns same data with new shape
    blocks = np.moveaxis(blocks, 1, 2)       # now blocks[i, j] has shape (sh, sw)
    ```

    Docs: `ndarray.reshape` / `numpy.reshape` and `numpy.moveaxis` ([NumPy][2]).
* **Template extraction** from **T_X**: use **row-major canonical cut** per input cell location:

  * For each ((i,j)), define the (sh × sw) **template from Π(X)** as:

    ```python
    # canonical: broadcast a single Π(X) pixel into (sh, sw) OR
    # (preferred) use Π(X) local patch repeated; here we fix the
    # “replicate center pixel of T_X[i, j]” as the template:
    template_v = np.full((sh, sw), T_X[i, j], dtype=T_X.dtype)
    ```
  * This “replicate the Π(X) type” is the minimal, unambiguous **types-only** template and matches 007bbfb7-style SBS (non-zero → one template, zero → another).
  * If you need a richer Π(X) template (e.g., using Π(X) *neighborhood*), you must anchor it explicitly in the addendum before changing this definition; for v0 we **only** replicate the Π(X) cell’s type.
* **Hashing**: `hashlib.sha256(template.tobytes()).hexdigest()` for receipts ([Python documentation][3]).

> No loops over pixels of the block to compare: assemble the `(sh, sw)` arrays and call `array_equal`. No stride tricks or homemade hashers.

---

## Verification logic (precise)

Let `(H, W) = X.shape`, `(h, w) = Y.shape`.
Compute `T_X = Π(X)` and `T_Y = Π(Y)` (WO-1).

1. **Divisible shape**
   If `h % H != 0 or w % W != 0`: return `None`.
   Else set `sh = h // H`, `sw = w // W`.

2. **Partition types**

   ```python
   blocks = T_Y.reshape(H, sh, W, sw)
   blocks = np.moveaxis(blocks, 1, 2)   # blocks[i, j] is (sh, sw)
   ```

3. **Build templates from Π(X) (canonical)**
   For each palette value `v` actually present in `X`:

   * Find the **first** `(i0, j0)` with `X[i0, j0] == v`.
   * Define the **template** for `v` as:

     ```python
     template_v = np.full((sh, sw), T_X[i0, j0], dtype=T_X.dtype)
     ```
   * For **every** `(i, j)` with `X[i, j] == v`, require:
     `np.array_equal(blocks[i, j], template_v)` → must be **True**; otherwise `return None`.

4. **σ table and hashes**

   * Compress templates into a deterministic list; build `sigma_table: {v -> template_id}` by the order the palette values appear (row-major in X).
   * `template_hashes[v] = sha256(template_v.tobytes())` for audit.

5. **Return**
   `("SBS-param", (sh, sw, sigma_table, template_hashes))`.

This is a strict, types-only SBS-Param proof. No colors; no reading quotas from inputs.

---

## Receipts (first-class)

Per training pair, append:

```json
{
  "task_id": "007bbfb7",
  "pair_index": 0,
  "free_sbs_param": {
    "H": 3, "W": 3, "h": 9, "w": 9,
    "sh": 3, "sw": 3,
    "palette_values_in_X": [0, 7],
    "sigma_table": {"0": 0, "7": 1},
    "template_hashes": { "0": "…", "7": "…" },
    "blocks_match_all": true
  },
  "candidate": ["SBS-param", [3, 3]]
}
```

On failure (any mismatch), emit the same structure with `blocks_match_all: false` and **no** `candidate`. Optionally include `{"v":7,"i":2,"j":0}` for the first failing block to speed reviews.

All booleans derive from one of the cited NumPy equality checks; hashes from `hashlib`. No heuristics.

---

## Runner changes

Add a receipts mode:

```bash
python -m arc.solve \
  --mode free-sbs-param-receipts \
  --challenges /mnt/data/arc-agi_training_challenges.json \
  --out outputs/receipts_wo3d.jsonl
```

Per training pair:

1. Call `types_from_output(X)` and `types_from_output(Y)` (WO-1) to get `T_X`, `T_Y`.
2. Call `verify_SBS_param(X, Y)`.
3. Emit the receipt line as above (JSONL; one object per line, UTF-8).

(If you prefer not to recompute Π(X) repeatedly, cache `T_X` per task; still call the same API.)

---

## Reviewer instructions (all 1000 tasks)

1. **Run:**

   ```bash
   python -m arc.solve \
     --mode free-sbs-param-receipts \
     --challenges /mnt/data/arc-agi_training_challenges.json \
     --out outputs/receipts_wo3d.jsonl
   ```

2. **Check:**

   * `(h,w)` equals `(sh*H, sw*W)` whenever a candidate exists.
   * `palette_values_in_X` matches the unique non-zero (and zero) symbols observed in `X`.
   * `blocks_match_all` is **true** iff a `candidate` is present; hash list size equals number of palette values seen.
   * Templates are **types** from Π(X) (hashes computed over `(sh, sw)` arrays filled by `T_X[i, j]` values), not colors.

3. **Gap vs bug:**

   * SBS-Param is either **proven** or **not**. There is no legit spec gap at this layer.
   * A **false positive** (candidate present but a block doesn’t equal its template) can only come from incorrect reshape/moveaxis or comparing color grids instead of types.
   * A **false negative** (should pass but returns None) typically indicates wrong canonical template definition or a bug in the equality loop.

---

## Performance guidance (CPU)

* Reshape/moveaxis and equality are vectorized in NumPy and fast on ARC sizes.
* Use exactly: `ndarray.reshape` / `numpy.reshape` ([NumPy][2]), `numpy.moveaxis` ([NumPy][4]), and `numpy.array_equal` ([NumPy][1]).
* No stride tricks or custom block walkers. No micro-optimizations.

---

## Acceptance criteria (green = WO-3D done)

* ✔ Runner completes across the corpus; one JSONL record per training pair.
* ✔ For every true SBS-Param pair, emits exactly one candidate with `(sh, sw)`, `sigma_table`, and `template_hashes`, and `blocks_match_all == true`.
* ✔ No false positives; deterministic receipts on re-run.
* ✔ Code `arc/free_sbs.py` + small runner plumbing; uses only the cited APIs.

---

[1]: https://numpy.org/doc/2.1/reference/generated/numpy.array_equal.html?utm_source=chatgpt.com "numpy.array_equal — NumPy v2.1 Manual"
[2]: https://numpy.org/doc/2.0/reference/generated/numpy.ndarray.reshape.html?utm_source=chatgpt.com "numpy.ndarray.reshape — NumPy v2.0 Manual"
[3]: https://docs.python.org/3/library/hashlib.html?utm_source=chatgpt.com "hashlib — Secure hashes and message digests"
[4]: https://numpy.org/devdocs/reference/generated/numpy.moveaxis.html?utm_source=chatgpt.com "numpy.moveaxis — NumPy v2.4.dev0 Manual"
