You’re right to block WO-7 until the parent mapping exists. This is not optional; it’s required by the math: each **new** disjointified type (S') must inherit the **paid** quotas from exactly one **parent** (S) (a Y₀ type). Below is a tight PATCH-WO for WO-5 to produce that mapping, with exact library calls and receipts so the reviewer can certify correctness at scale.

---

# PATCH-WO (WO-5) — Return `parent_of` mapping from Disjointify

## Anchors

* `00_math_spec.md` §5 (FREE transport on **types**; lawful composition A2).
* `00_math_spec.md` §4 (per-type quotas are **paid** on Y₀ types; must apply to the same types’ descendants in test).
* `01_math_spec_addendum.md` (replicated copies must be **disjoint**; fills cannot bleed across blocks).

## API changes (frozen)

```python
# arc/transport.py

from typing import Tuple, Dict, Any
import numpy as np

def transport_types(
    T_train: np.ndarray,                           # Π(Y0) types
    free_tuple: Tuple[str, Tuple[Any, ...]],
    X_test_shape: Tuple[int, int],
    X_test: np.ndarray | None
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Build T_test per the proven FREE terminal (types-only), then disjointify.
    Returns:
        T_test_disjoint : np.ndarray[int]  # final test mosaic after 4-conn split
        parent_of       : Dict[int, int]   # new type id S' -> parent Y0 type id S
    """
```

## Implementation plan (mature libs, no custom algos)

1. **Build the pre-disjoint test mosaic** `T_pre` exactly as you already do for the chosen terminal:

   * `identity`, `h/v concat-dup`, `h-mirror-concat`: `numpy.concatenate`, `numpy.fliplr`. ([NumPy][1])
   * `tile (sh,sw)`: `numpy.tile` or `numpy.take(..., mode="wrap")`. ([NumPy][2])
   * `SBS-*`: write each `(sh,sw)` block into `T_pre` using types-only templates.

2. **Disjointify by 4-connected components per parent type**
   Use `skimage.measure.label(mask, connectivity=1)` for each parent type id `S` present in `T_pre` (mask = `T_pre == S`). This returns a labeled array where each connected region (4-conn) has a unique positive id; doc: “labels ndarray… connected regions are assigned the same integer value.” ([Scikit-image][3])
   Why per-type? It guarantees we never merge components across different parent types.

3. **Assign dense new ids and build the mapping**

   * Maintain a global `next_id = 0`.
   * For each `S` (iterate in **ascending** numeric order for determinism):

     * `labels = measure.label(T_pre == S, connectivity=1)` (same shape as `T_pre`).
     * For each `lab` in `np.unique(labels)` excluding 0:

       * Set `T_disjoint[labels == lab] = next_id`
       * Record `parent_of[next_id] = S`
       * `next_id += 1`
   * This yields a **dense** id space `0..next_id-1`, stable across runs.

4. **Return** `(T_disjoint, parent_of)`.

### Notes on the exact calls

* `skimage.measure.label` arguments: `connectivity=1` gives 4-connectivity in 2D; the function returns the labeled array and (optionally) the number of labels. ([Scikit-image][3])
* If you want an official example of labeling usage, see scikit-image docs “Label image regions.” ([Scikit-image][4])
* If `skimage` isn’t available, you can temporarily fall back to NumPy+DFS, but the preferred path is `skimage.measure.label` to avoid reinventing CCL algorithms.

## Receipts (first-class; emitted by WO-5 after the patch)

Per FREE_PROVEN task, add a `disjointify` section:

```json
{
  "task_id": "…",
  "disjointify": {
    "pre":  { "unique_types":  K_pre },   // |unique(T_pre)|
    "post": { "unique_types":  K_post },  // |unique(T_disjoint)|
    "parent_map": {
      "size": K_post,
      "samples": [ ["0","7"], ["1","7"], ["2","12"] ] // [S', S] few pairs as strings
    },
    "parent_consistency_pass": true,      // every S' maps to exactly one S and T_pre==S for all its pixels
    "counts_conserved_pass":   true       // sum over S' children sizes == size of their parent mask in T_pre
  }
}
```

How to compute the two booleans, entirely with NumPy:

* **parent_consistency_pass**: for each `S'`, compute `idx = np.flatnonzero(T_disjoint.ravel() == S')` (doc) ([NumPy][2]), reshape back to 2D with `ravel()[idx]` indices if needed, then assert `np.all(T_pre.ravel()[idx] == parent_of[S'])`.
* **counts_conserved_pass**: for each parent `S`, check `sum_{S' : parent_of[S']=S} |{T_disjoint == S'}| == |{T_pre == S}|`. Use `np.unique` with counts or `np.count_nonzero` to tally sizes. ([NumPy][5])

(You can also include `sha256` of `T_disjoint` for integrity; same hashing approach we use elsewhere.)

## Runner: small change

* Update `--mode transport-receipts` (WO-5) so it **returns and logs** the new `parent_of` map. Persist it in the receipts JSONL (as above) so WO-7 can consume it deterministically.

## Reviewer: what to check (all FREE_PROVEN tasks)

* `post.unique_types ≥ pre.unique_types`.
* `parent_consistency_pass == true` and `counts_conserved_pass == true` for **every** task.
* Spot-check a few mappings by directly verifying `T_pre == S` on the pixels of a reported `S'`.

## Why this is required (math-level)

* Quotas (K_{S,\cdot}) are **paid** on Y₀ types (S). After transport and disjointify, each new block (S') must **inherit** those quotas from exactly one (S); otherwise the meet step can’t be defined unambiguously. This patch provides that exact witness function (S' \mapsto S).

## Complexity

* For each unique parent type (S), one `label` call and a few NumPy assignments. On ARC sizes, total is O(pixels) per task.

---

### Citations for the exact primitives we’re invoking

* **`skimage.measure.label` (connectivity=1, returns labeled array)**: official API. ([Scikit-image][3])
* **`numpy.unique` with counts**: official API. ([NumPy][1])
* **`numpy.flatnonzero`**: official API. ([NumPy][2])

---

### Acceptance (green for the patch)

* `transport_types(...)` now returns `(T_disjoint, parent_of)`.
* Receipts include `parent_map.size == post.unique_types`.
* `parent_consistency_pass == true` and `counts_conserved_pass == true` for all **transported** tasks.
* Deterministic across runs.

[1]: https://numpy.org/doc/stable/reference/generated/numpy.unique.html?utm_source=chatgpt.com "numpy.unique — NumPy v2.3 Manual"
[2]: https://numpy.org/devdocs/reference/generated/numpy.flatnonzero.html?utm_source=chatgpt.com "numpy.flatnonzero — NumPy v2.4.dev0 Manual"
[3]: https://scikit-image.org/docs/0.25.x/api/skimage.measure.html?utm_source=chatgpt.com "skimage.measure — skimage 0.25.2 documentation"
[4]: https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_label.html?utm_source=chatgpt.com "Label image regions — skimage 0.24.0 documentation"
[5]: https://numpy.org/doc/2.1/reference/generated/numpy.unique.html?utm_source=chatgpt.com "numpy.unique — NumPy v2.1 Manual"
