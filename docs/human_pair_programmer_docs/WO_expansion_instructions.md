now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for
## WO-6 — Quotas K (Paid) + Y₀ Selection

**Goal:** Count per-type color quotas from one training output; deterministic Y₀ policy.

* **Scope:** Select Y₀ whose non-zero palette matches the test’s, tie-break lexicographically, else first. Compute K on Π(Y₀). (Type alignment across Y_i not required in v0.)
* **Libs:** `numpy`
* **IO:** `choose_Y0(task) -> Y0_idx`, `quotas(Y0, T0, C) -> K`
* **Receipts:** Y₀ selection reason; per-type K with ∑K==|S|; palette report.
* **Pass criteria:** All tasks produce valid K; no quotas from inputs.

now here are the things u must take care of:
1. now thr is nothing called underspecificy in dev. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created.
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. then we must hv receipts as first class citizens part of WO with clear understanding as to how they capture and ensure that our implementation matches math spec 1:1.
5. make sure u include runner changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests 
6. a clear instruction to reviewer that tells them how to use runner for all 1000 arc agi tasks, how to use receipts to knw that math implementation is right.. and how to identtify a legit implementation gap or unsatisifable task if any in WO. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?