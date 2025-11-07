now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for
## WO-5 — Transport Types + Disjointify

**Goal:** Build the **test** type mosaic exactly as the proven FREE map dictates; keep copies disjoint.

* **Scope:** Implement transport for terminals from WO-3/4: identity, mirror-concat, v-double, concat-dup, tile, SBS-Y, SBS-param. Then disjointify (connected-component relabel) so fills don’t bleed.
* **Libs:** `numpy`, `skimage.measure.label` (4-conn) or simple DFS
* **IO:** `transport_types(T_train, free_tuple, X_test_shape, X*, Y0) -> T_test`
* **Receipts:** output shape; per-block template hash match; pre/post disjoint type counts.
* **Pass criteria:** All FREE_PROVEN tasks pass shape checks and block template matches.

now here are the things u must take care of:
1. now thr is nothing called underspecificy in dev. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created.
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. then we must hv receipts as first class citizens part of WO with clear understanding as to how they capture and ensure that our implementation matches math spec 1:1.
5. make sure u include runner changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests 
6. a clear instruction to reviewer that tells them how to use runner for all 1000 arc agi tasks, how to use receipts to knw that math implementation is right.. and how to identtify a legit implementation gap or unsatisifable task if any in WO. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?