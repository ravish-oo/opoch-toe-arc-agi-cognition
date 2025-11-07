now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for
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

now here are the things u must take care of:
1. now thr is nothing called underspecificy in dev. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created.
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. then we must hv receipts as first class citizens part of WO with clear understanding as to how they capture and ensure that our implementation matches math spec 1:1.  pls ensure that the receipts flow from this WO-03xx wo to WO-04's file and eventually to solve.py to keep our testing and debugging pipeline tight and intact 
5. make sure u include runner changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests 
6. a clear instruction to reviewer that tells them how to use runner for all 1000 arc agi tasks, how to use receipts to knw that math implementation is right.. and how to identtify a legit implementation gap or unsatisifable task if any in WO. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)
8. Ensure that u add explicit wiring instruction to WO-04 so that we can test immediately if it increases some matches

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?

=======

now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for
## WO-8 — v0 Runner (Batch + Audit)

**Goal:** End-to-end solve on all tasks we can prove FREE for; write predictions and receipts.

* **Scope:** `solve_batch` that: Π on Y₀ → K; FREE proof → transport → fill; write outputs JSON and receipts JSONL; `audit <id>` CLI to dump receipts for one task.
* **Libs:** `numpy`, `typer/argparse`, `hashlib`
* **IO:** `solve(challenges.json) -> predictions.json + receipts.jsonl`
* **Receipts:** corpus summary (counts by terminal, FREE_PROVEN vs UNPROVEN), frozen simplicity order echo.
* **Pass criteria:** **600+ tasks** attemptable now; outputs + receipts produced deterministically; no crashes on full corpus.

now here are the things u must take care of:
1. now thr is nothing called underspecificy in dev. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created.
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. then we must hv receipts as first class citizens part of WO with clear understanding as to how they capture and ensure that our implementation matches math spec 1:1.
5. make sure u include runner changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests 
6. a clear instruction to reviewer that tells them how to use runner for all 1000 arc agi tasks, how to use receipts to knw that math implementation is right.. and how to identtify a legit implementation gap or unsatisifable task if any in WO. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?