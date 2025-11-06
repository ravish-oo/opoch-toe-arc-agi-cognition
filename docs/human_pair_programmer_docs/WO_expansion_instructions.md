now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for
## WO-4 — FREE Intersection + Pick (Frozen Order)

**Goal:** Task-level proof: intersect per-pair candidates and select terminal per frozen order.

* **Scope:** Intersect candidates across all train pairs **per slot**; prefer no D4/translate (slots included but default “none” in v0); select terminal by fixed order:

  1. identity, 2) {h-mirror, v-double, h/v-dup}, 3) tile, 4) SBS-Y, 5) SBS-param.
* **Libs:** `numpy`
* **IO:** `prove_free(task) -> ("FREE_PROVEN", tuple) | ("FREE_UNPROVEN", reason)`
* **Receipts:** list all candidates per pair, the intersected set, and the chosen terminal with parameters.
* **Pass criteria:** On the full corpus, produces a proven tuple for ~**600+** tasks (the “attemptable now” bucket) and marks others unproven; no ad-hoc fallbacks.

now here are the things u must take care of:
1. now thr is nothing called underspecificy in dev. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created.
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. then we must hv receipts as first class citizens part of WO with clear understanding as to how they capture and ensure that our implementation matches math spec 1:1.
5. make sure u include runner changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests 
6. a clear instruction to reviewer that tells them how to use runner for all 1000 arc agi tasks, how to use receipts to knw that math implementation is right.. and how to identtify a legit implementation gap or unsatisifable task if any in WO. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?