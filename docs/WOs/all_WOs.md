Locked. Here are the **v0 high-level Work Orders** — bottoms-up, atomic, self-contained, and strictly spec+addendum — that together make **600+ tasks attemptable**. Each WO ends in receipts the reviewer can run on all 1000 tasks.

---

## WO-1 — Π Ruler (Types) ✅ COMPLETED

**Goal:** Implement the fixed, idempotent Π typing on any grid.

* **Scope:** φ stencil (center; N/E/S/W; 2-step ring; edge flags; parity; sentinel −1). Hash φ→type ids. Deterministic ordering.
* **Libs:** `numpy`
* **IO:** `types_from_output(Y) -> (T, codebook)`
* **Receipts:** idempotence (hash(Π(Y)) == hash(Π(Π(Y)))), codebook hash stability, partition totals ∑|S| = H·W.
* **Pass criteria:** All 1000 tasks pass receipts A1–A3 (idempotence, stability, totals).

---

## WO-2 — Harness + Receipts Skeleton ✅ COMPLETED

**Goal:** Corpus runner and per-task receipts plumbing, *no FREE yet*.

* **Scope:** Load ARC JSON, iterate tasks, run WO-1 on all training outputs and cache Π receipts.
* **Libs:** `numpy`, `hashlib`, minimal CLI (`typer` or `argparse`)
* **IO:** `run_receipts(tasks) -> receipts.jsonl` (per-task Π receipts), summary print.
* **Receipts:** Writes Π receipts only; counts tasks processed.
* **Pass criteria:** Runs on all 1000 without error; receipts present for each train Y; Π invariants hold globally.

---

## WO-3A — FREE Verifiers: Identity / Mirror-Concat / V-Double / Concat-Dup ✅ COMPLETED

**Goal:** Per training pair, verify “simple” terminals exactly (color-level equality where defined).

* **Scope:**

  * `identity` (shape equal),
  * `h-mirror-concat` (rev|id, id|rev),
  * `v-double`,
  * `h-concat-dup`, `v-concat-dup`.
* **Libs:** `numpy`
* **IO:** `verify_simple_free(X,Y) -> List[(kind, params)]` per pair.
* **Receipts:** for each pair, emit the candidate set; for each task, union of per-pair candidates.
* **Pass criteria:** Runs on all 1000; no heuristics; receipts list exactly what matched.

---

## WO-3B — FREE Verifier: Types-Periodic Tile ✅ COMPLETED

**Goal:** Detect integer blow-up by **types** periodicity (FREE).

* **Scope:** Verify `tile (sh,sw)` by checking `T_Y[r,c] == T_Y[r%H, c%W]`.
* **Libs:** `numpy`
* **IO:** `verify_tile_types(X,Y,T_Y) -> Optional(("tile",(sh,sw)))`
* **Receipts:** record `(sh,sw)` and a boolean “types periodicity” proof.
* **Pass criteria:** No false positives on corpus; receipts present when true.

---

## WO-3C — FREE Verifier: SBS-Y (Selector-Driven Block Substitution from Π(Y)) ✅ COMPLETED

**Goal:** Prove SBS using **templates from training output types**.

* **Scope:** For h/H = sh, w/W = sw; for each block (i,j), assert `t_Y[i*sh:(i+1)*sh, j*sw:(j+1)*sw] == B^{σ(X[i,j])}` where templates (B^{(m)}) come from Π(Y) blocks; σ is finite on palette.
* **Libs:** `numpy`
* **IO:** `verify_SBS_Y(X, T_Y) -> Optional(("SBS-Y", (sh,sw, σ_table, template_hashes)))`
* **Receipts:** `(sh,sw)`, σ table, per-template type hashes; per-block match booleans.
* **Pass criteria:** Strict match only; emits full proof objects.

---

## WO-3D — FREE Verifier: SBS-Param (Templates from Π(X)) ✅ COMPLETED

**Goal:** Prove SBS using **templates from input types** (the 007bbfb7 fix).

* **Scope:** Same as 3C but templates are `Π(X)` 3×3 blocks; σ(x)=1_{x≠0} or finite table; verify blocks of `t_Y` equal templates chosen by σ.
* **Libs:** `numpy`
* **IO:** `verify_SBS_param(X, Y) -> Optional(("SBS-param", (sh,sw, σ_table, template_hashes_of_ΠX)))`
* **Receipts:** Same structure as 3C, but logs Π(X) template hashes.
* **Pass criteria:** Strict; no quotas from inputs (types only).

---

## WO-3E — FREE Verifiers: **D4 Isometries** + **Integer Translations**

**Scope (types only)**
For each train pair ((X!\to Y)), verify:

* `D4(rot90|rot180|rot270|flip_h|flip_v|flip_h+rot90|flip_h+rot270)`: prove `T_Y == d4(T_X)` where `d4` uses `np.rot90`/`np.fliplr`/`np.flipud` on **types** (not colors) ([NumPy][1]).
* `translate(dr, dc)`: prove `T_Y == np.roll(T_X, shift=(dr, dc), axis=(0,1))` (pure wrap translation) on **types** ([NumPy][2]).

**Libs & calls (no custom algos)**
`numpy.rot90` (doc) ([NumPy][1]), `numpy.fliplr`/`flipud`, `numpy.roll` (doc) ([NumPy][2]), `numpy.array_equal`.

**IO**
`verify_d4_and_translate(T_X, T_Y) -> List[(kind, params)]`

**Receipts**
For each candidate: record `kind`, `params`, and boolean `array_equal(T_Y, transform(T_X))`.

**Pass**
Runs on all pairs, no false positives.

**Runner wiring**
Only **WO-4** consumes these candidates. No direct use in `solve.py`.

---

## WO-3F — FREE Verifiers: **H/V concat** + **Banded double** (types only)

**Scope (types only)**
Per pair, verify **composition-free** duplications on **types**:

* `h-concat-dup`: (T_Y = [T_X\ |\ T_X]) via `np.concatenate([T_X, T_X], axis=1)` (documented) ([NumPy][3]).
* `v-concat-dup`: (T_Y = [T_X ; T_X]) via `np.concatenate([T_X, T_X], axis=0)` ([NumPy][3]).
* `banded v-double / h-double` (optional v1): prove (T_Y) equals two stacked/side-by-side **bands** of (T_X) without mixing types across the band boundary (implemented by slicing and concatenation; mature `np.concatenate`).

**Libs & calls**
`numpy.concatenate` (doc) ([NumPy][3]), `numpy.array_equal`.

**IO**
`verify_concat_banded(T_X, T_Y) -> List[(kind, params)]` (params can be `None` or band sizes).

**Receipts**
Record shapes and `array_equal` checks of subblocks.

**Pass**
No false positives.

**Runner**
WO-4 only.

---

## WO-3B/3C/3D **(Already done)** — **Integer tile** + **SBS-Y** + **SBS-Param**

* Tile(types) uses `np.tile` or `np.take(..., mode='wrap')` on **types** (docs) ([scikit-image.org][4]).
* SBS-Y/Param use `reshape` + `moveaxis` on **types** to partition blocks; compare with `array_equal` (docs for `moveaxis`) ([NumPy][5]).

*No change here; listed for completeness.*

---

## WO-3G — FREE Verifier: **Component-wise transport (per-component pose, types only)**

**Scope (types only)**
Per pair, detect **connected components** in (T_X) and prove that each component is carried to a corresponding set of components in (T_Y) via a **single global pose** (composition of a D4 and a translation).

1. Label components in (T_X) and (T_Y) with **4-connectivity** using `skimage.measure.label` (documented; `connectivity=1`) ([scikit-image.org][4]).
2. For each labeled component in (T_X), extract its type mask and bounding box; apply each D4 pose and integer translations (`np.rot90`, `np.roll`) and verify exact equality to **some** union of labeled components in (T_Y) (types only) ([NumPy][1]).
3. Accept if a **single pose** explains all components (same D4 and same translation per component relative to its bbox origin).

**Libs & calls**
`skimage.measure.label` (4-conn) ([scikit-image.org][4]), `numpy.rot90` ([NumPy][1]), `numpy.roll` ([NumPy][2]), `numpy.array_equal`.

**IO**
`verify_component_transport(T_X, T_Y) -> Optional(("component_transport", (d4, dr, dc)))`

**Receipts**

* Component counts in (T_X, T_Y); per component: bbox, applied pose, boolean match.
* `components_covered = 100%` and `single_pose=True`.

**Pass**
No false positives; returns one terminal if every (X) component is explained by the same pose.

**Runner**
WO-4 consumes this terminal at **lower priority** than `identity` if you keep the frozen order or **insert** after D4/translate to reduce “identity” dominance (you can append to the order without regressing v0).

---

## WO-3H — FREE Verifier: **Stripe / Column / Row projections** (types only)

**Scope (types only)**
Per pair, prove that (T_Y) is obtained by selecting/stacking rows or columns of (T_X) with a **fixed pattern** (e.g., 3×7→3×3 left/right fold; stride-2 select; interleave).

* Implement patterns as **index maps** over axes using `np.take(indices, axis=?, mode='wrap')` (doc) and/or simple slicing; compare with `array_equal` on **types** ([NumPy][6]).

**Libs & calls**
`numpy.take(mode='wrap')` (doc) ([NumPy][6]), `numpy.array_equal`.

**IO**
`verify_stripe_maps(T_X, T_Y) -> Optional(("band_map", pattern_spec))`
`pattern_spec` is a small tuple: (axis, indices or stride, fold rule).

**Receipts**
Record `(axis, indices/stride)`, `array_equal` true, and the number of selected rows/cols.

**Pass**
No false positives; patterns must be **constant across all rows/cols**.

**Runner**
WO-4 consumes this terminal; append it **after** concat/doubles and before tile/SBS to tighten structure early.

---

## WO-3I — FREE Verifier: **Type-channel permutations** (types only)

**Scope (types only)**
Some ARC demos “rename” types (permutation of Π-ids) without changing geometry. Verify that (T_Y) is a **global relabeling** of (T_X): (T_Y = P(T_X)) for a **single permutation** (P) of type ids.

* Build mapping by pairing sorted type masks (e.g., lexicographically encode each type mask by `np.flatnonzero` indices) and verify **one-to-one** mapping such that `T_Y == P(T_X)`.

**Libs & calls**
`numpy.flatnonzero`, `numpy.array_equal`.

**IO**
`verify_type_perm(T_X, T_Y) -> Optional(("type_perm", mapping_hash))`

**Receipts**

* Number of types matched, a hash of the mapping, and boolean `array_equal(T_Y, P(T_X))`.

**Pass**
No false positives (requires a global bijection).

**Runner**
WO-4 consumes this (place it with lowest priority; it’s rarely needed but harmless to check).

---

## WO-4 — FREE Intersection + Pick (Frozen Order) ✅ COMPLETED

**Goal:** Task-level proof: intersect per-pair candidates and select terminal per frozen order.

* **Scope:** Intersect candidates across all train pairs **per slot**; prefer no D4/translate (slots included but default “none” in v0); select terminal by fixed order:

  1. identity, 2) {h-mirror, v-double, h/v-dup}, 3) tile, 4) SBS-Y, 5) SBS-param.
* **Libs:** `numpy`
* **IO:** `prove_free(task) -> ("FREE_PROVEN", tuple) | ("FREE_UNPROVEN", reason)`
* **Receipts:** list all candidates per pair, the intersected set, and the chosen terminal with parameters.
* **Pass criteria:** On the full corpus, produces a proven tuple for ~**600+** tasks (the “attemptable now” bucket) and marks others unproven; no ad-hoc fallbacks.

---

## WO-5 — Transport Types + Disjointify ✅ COMPLETED

**Goal:** Build the **test** type mosaic exactly as the proven FREE map dictates; keep copies disjoint.

* **Scope:** Implement transport for terminals from WO-3/4: identity, mirror-concat, v-double, concat-dup, tile, SBS-Y, SBS-param. Then disjointify (connected-component relabel) so fills don’t bleed.
* **Libs:** `numpy`, `skimage.measure.label` (4-conn) or simple DFS
* **IO:** `transport_types(T_train, free_tuple, X_test_shape, X*, Y0) -> T_test`
* **Receipts:** output shape; per-block template hash match; pre/post disjoint type counts.
* **Pass criteria:** All FREE_PROVEN tasks pass shape checks and block template matches.

---

## WO-6 — Quotas K (Paid) + Y₀ Selection ✅ COMPLETED

**Goal:** Count per-type color quotas from one training output; deterministic Y₀ policy.

* **Scope:** Select Y₀ whose non-zero palette matches the test’s, tie-break lexicographically, else first. Compute K on Π(Y₀). (Type alignment across Y_i not required in v0.)
* **Libs:** `numpy`
* **IO:** `choose_Y0(task) -> Y0_idx`, `quotas(Y0, T0, C) -> K`
* **Receipts:** Y₀ selection reason; per-type K with ∑K==|S|; palette report.
* **Pass criteria:** All tasks produce valid K; no quotas from inputs.

---

## WO-7 — Fill by Rank + Idempotence ✅ COMPLETED

**Goal:** Produce (Y^*) via the meet rule and certify fixed-point behavior.

* **Scope:** Row-major ranks within each transported type; cumulative cuts Σ; assign smallest c with rank ≤ Σ(c); recompute Π/K on (Y^*), re-fill to confirm idempotence.
* **Libs:** `numpy`
* **IO:** `fill_by_rank(T_test, K, C) -> Y*`
* **Receipts:** quota satisfaction per type (counts match K); per-rank minimality; idempotence true; SHA256(Y*).
* **Pass criteria:** All FREE_PROVEN tasks satisfy quotas and idempotence; hashes emitted.

---

## WO-8 — v0 Runner (Batch + Audit) ✅ COMPLETED

**Goal:** End-to-end solve on all tasks we can prove FREE for; write predictions and receipts.

* **Scope:** `solve_batch` that: Π on Y₀ → K; FREE proof → transport → fill; write outputs JSON and receipts JSONL; `audit <id>` CLI to dump receipts for one task.
* **Libs:** `numpy`, `typer/argparse`, `hashlib`
* **IO:** `solve(challenges.json) -> predictions.json + receipts.jsonl`
* **Receipts:** corpus summary (counts by terminal, FREE_PROVEN vs UNPROVEN), frozen simplicity order echo.
* **Pass criteria:** **600+ tasks** attemptable now; outputs + receipts produced deterministically; no crashes on full corpus.

---

These eight WOs are small, additive, and testable from day one. They stick 100% to spec + addendum (including SBS-param), and they get us to the 600+ attemptable milestone cleanly. When v0 is green, we can add `band_map` and `component_transport` later as new WOs without touching the ones above.
