# WO-3C — FREE Verifier: SBS-Y (Selector-Driven Block Substitution from Π(Y))

## Anchors to read first

* `docs/anchors/00_math_spec.md` — §5 *Shape Map (Free Transport)*: fixed FREE maps verified from a training pair, with proofs done on **types**.
* `docs/anchors/01_math_spec_addendum.md` — SBS definition: block grid of size ((s_h,s_w)); per-block template chosen only by the input cell via a finite selector (\sigma); **templates are Π(Y) type blocks**; quotas are **not** read from inputs.

## Goal

For a training pair ((X!\to Y)) with (X\in\mathcal C^{H\times W}, Y\in\mathcal C^{h\times w}) and a precomputed type mosaic (T_Y=\Pi(Y)), **prove SBS-Y** as a FREE map by showing:

1. (h= s_h\cdot H) and (w=s_w\cdot W) for integers (s_h, s_w).
2. When you partition (T_Y) into a non-overlapping grid of (H\times W) **blocks** of shape ((s_h,s_w)), then for each input value (v\in\mathcal C), all blocks at positions ((i,j)) where (X_{ij}=v) are **exactly equal** (same type pattern); call this template (B^{(v)}).
3. This builds a finite selector (\sigma:\mathcal C\to{m}) (palette value → template id), with the **templates drawn from Π(Y) blocks**.

If all conditions pass, return a single candidate:

```python
("SBS-Y", (sh, sw, sigma_table, template_hashes))
```

where:

* `sh = h//H`, `sw = w//W`
* `sigma_table`: dict {int palette_value -> int template_id (0..M-1)} (only for values that appear in X)
* `template_hashes`: list of sha256 hex digests of each template’s **type** block bytes (for audit)

Strict equality; no heuristics; no color-level checks.

---

## Interface (frozen)

```python
# arc/free_sbs.py
from typing import Optional, Tuple, Dict, Any
import numpy as np

def verify_SBS_Y(X: np.ndarray, T_Y: np.ndarray) \
    -> Optional[Tuple[str, Tuple[int, int, Dict[int, int], Dict[int, str]]]]:
    """
    Prove SBS-Y on types. Return:
      ("SBS-Y", (sh, sw, sigma_table, template_hashes))
    or None if not proven.
    """
```

* Deterministic output; return **None** or exactly one tuple.
* This verifier operates **only on types** (Π(Y) given as `T_Y`). No color reads from X or Y.

---

## Exact libraries & functions to use (no custom algos)

Use only well-documented NumPy and stdlib:

* **Block reshaping** (non-overlapping grid):
  Since (h=s_h H) and (w=s_w W), reshape the type mosaic:

  ```python
  # T_Y shape (h, w) -> (H, sh, W, sw) then move axes to (H, W, sh, sw)
  blocks = T_Y.reshape(H, sh, W, sw)         # ndarray.reshape is standard
  blocks = np.moveaxis(blocks, 1, 2)         # so blocks[i,j] has shape (sh, sw)
  ```

  `ndarray.reshape` is officially documented and preserves data with a new shape; this is a view or copy as permitted by NumPy’s rules ([numpy.org][1]).
  (You may also use `numpy.reshape` free function equivalently ([numpy.org][2]).)

* **Exact equality** between blocks and templates: `numpy.array_equal(a, b)` (no tolerance) ([numpy.org][3]).

* **Templating by first occurrence:** for each palette value (v) seen in `X`, pick the **first** block (`i,j`) where `X[i,j]==v` as the template, then require all other `(i,j)` with the same (v) to satisfy `array_equal(blocks[i,j], template)`.

* **Hashing templates for receipts:** `hashlib.sha256(block.tobytes()).hexdigest()` (stable, deterministic) ([Python documentation][4]).

> **Do not** use stride tricks like `as_strided` in v0 (error-prone). Standard `reshape`/`moveaxis` is sufficient and clear.

---

## Verification logic (precise, step-by-step)

Let `(H, W) = X.shape`, `(h, w) = T_Y.shape`.

1. **Divisibility check**
   If `h % H != 0 or w % W != 0`: return `None`.
   Else set `sh = h//H`, `sw = w//W`.

2. **Partition into blocks (types only)**

   ```
   # preconditions hold; produce a (H, W, sh, sw) view
   blocks = T_Y.reshape(H, sh, W, sw)
   blocks = np.moveaxis(blocks, 1, 2)   # now blocks[i,j] is (sh, sw)
   ```

3. **Build σ and templates**

   * Initialize: `sigma_table = {}`, `templates = {}`.
   * For each value `v` in the set of values appearing in `X` (row-major order over `i,j`):

     * Find the first `(i,j)` with `X[i,j]==v` → set `templates[v] = blocks[i,j]`.
     * For every other `(i,j)` with `X[i,j]==v`, require `np.array_equal(blocks[i,j], templates[v])` to be `True`. If any is `False`, return `None`. ([numpy.org][3])
   * Compress `templates` to a list and map `v → template_id` to produce `sigma_table`.

4. **Return proof object**

   * `template_hashes[v] = sha256(templates[v].tobytes())` (hex) for audit ([Python documentation][4])
   * `return ("SBS-Y", (sh, sw, sigma_table, template_hashes))`

No other checks. This is a strict SBS-Y proof on types.

---

## Receipts (first-class)

For **each training pair**, append a JSON object to the receipts stream:

```json
{
  "task_id": "007bbfb7",
  "pair_index": 0,
  "free_sbs_y": {
    "H": 3, "W": 3, "h": 9, "w": 9,
    "sh": 3, "sw": 3,
    "palette_values_in_X": [0,7],
    "sigma_table": {"0": 0, "7": 1},
    "template_hashes": {"0": "…", "7": "…"},
    "blocks_match_all": true        // every (i,j) with same X[i,j] had identical (sh,sw) type blocks
  },
  "candidate": ["SBS-Y", [3, 3]]
}
```

If the proof fails (e.g., a mismatch for some `v`), emit:

```json
{
  "task_id": "…", "pair_index": 0,
  "free_sbs_y": {
    "H":…, "W":…, "h":…, "w":…,
    "sh":…, "sw":…,
    "palette_values_in_X": [ … ],
    "blocks_match_all": false,
    "mismatch_example": {"v":7, "i":2, "j":0}   // first failing location (optional)
  }
}
```

Every boolean is an exact NumPy equality; no scores or thresholds.

---

## Runner changes

Add a receipts mode to the CLI:

```bash
python -m arc.solve \
  --mode free-sbs-y-receipts \
  --challenges /mnt/data/arc-agi_training_challenges.json \
  --out outputs/receipts_wo3c.jsonl
```

Behavior per training pair in each task:

1. Compute `T_Y = arc.pi.types_from_output(Y)[0]` (Π from WO-1).
2. Call `verify_SBS_Y(X, T_Y)`.
3. Emit the per-pair receipt above (include `candidate` only when proven).

As with earlier WOs, write **JSON Lines** (one JSON object per line, UTF-8, newline-delimited). JSONL is line-oriented; do not batch records into a single JSON array. ([numpy.org][5])

---

## Reviewer instructions (all 1000 tasks)

1. **Run**:

   ```bash
   python -m arc.solve \
     --mode free-sbs-y-receipts \
     --challenges /mnt/data/arc-agi_training_challenges.json \
     --out outputs/receipts_wo3c.jsonl
   ```

2. **Check**:

   * For each pair, when `(h,w) == (sh*H, sw*W)`, verify `blocks_match_all == true` accompanies a `candidate = ["SBS-Y",[sh,sw]]`.
   * Confirm that `sigma_table` covers exactly the palette values seen in `X` (reported in `palette_values_in_X`).
   * Templates are **types**, not colors (hashes were computed from `blocks[i,j]`, drawn from `T_Y`, not `Y`).

3. **Identify a legit gap vs. bug**:

   * SBS-Y is either **proven** or not. There is no “legit spec gap” at this layer: a failing proof means the pair does **not** support SBS-Y, or the implementation is wrong.
   * A **false positive** is unacceptable; it would imply `array_equal` succeeded erroneously. That points to a coding bug (wrong reshape, axis order, or compared color grid instead of types).
   * A **false negative** (missed SBS-Y) typically indicates the reshape grid wasn’t `(H, sh, W, sw) → (H, W, sh, sw)` or that equality wasn’t checked against a stable “first template” for each `v`.

---

## Performance guidance (CPU)

* Reshape + moveaxis + equality checks are vectorized and fast in NumPy; ARC grids are small. No optimizations or stride tricks required.
* You **may** use `numpy.lib.stride_tricks.sliding_window_view` if you want a windowed view, but it’s unnecessary here and more complex; stick to `reshape`/`moveaxis` for clarity ([numpy.org][6]).

---

## Acceptance criteria (green = WO-3C done)

* ✔ Runner completes over the full corpus; writes one JSONL record per training pair.
* ✔ For every true SBS-Y pair, emits exactly one candidate with `(sh,sw)`, `sigma_table`, and `template_hashes`, and sets `blocks_match_all=true`.
* ✔ No false positives; deterministic receipts across runs.
* ✔ Code `arc/free_sbs.py` + small runner plumbing; uses only `reshape`, `moveaxis`, `array_equal`, and `hashlib.sha256`.

---

[1]: https://numpy.org/doc/2.0/reference/generated/numpy.ndarray.reshape.html?utm_source=chatgpt.com "numpy.ndarray.reshape — NumPy v2.0 Manual"
[2]: https://numpy.org/devdocs/reference/generated/numpy.reshape.html?utm_source=chatgpt.com "numpy.reshape — NumPy v2.4.dev0 Manual"
[3]: https://numpy.org/doc/2.1/reference/generated/numpy.array_equal.html?utm_source=chatgpt.com "numpy.array_equal — NumPy v2.1 Manual"
[4]: https://docs.python.org/3/library/hashlib.html?utm_source=chatgpt.com "hashlib — Secure hashes and message digests"
[5]: https://numpy.org/devdocs/reference/generated/numpy.take.html?utm_source=chatgpt.com "numpy.take — NumPy v2.4.dev0 Manual"
[6]: https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html?utm_source=chatgpt.com "lib.stride_tricks.sliding_window_view"
