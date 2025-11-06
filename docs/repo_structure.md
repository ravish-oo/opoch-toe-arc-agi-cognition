Here’s a lean, unambiguous repo layout. It’s minimal, v0-focused, and matches the spec+addendum and our 8 WOs.

```text
.
├── README.md
├── pyproject.toml            # deps: numpy, scikit-image, typer (or argparse), pytest
├── requirements.txt          # optional alt to pyproject
├── docs/
│   ├── anchors/
│   │   ├── 00_math_spec.md
│   │   └── 01_math_spec_addendum.md
│   └── repo_structure.md     # this file
├── arc/                      # all implementation; ≤500 LOC per module
│   ├── __init__.py
│   ├── pi.py                 # WO-1: Π ruler (types_from_output, codebook, idempotence helpers)
│   ├── receipts.py           # WO-2: receipts schema + writer (JSONL), SHA256, helpers
│   ├── free_simple.py        # WO-3A+3B: identity, mirror-concat, v-double, concat-dup, tile(types)
│   ├── free_sbs.py           # WO-3C+3D: SBS-Y and SBS-param verifiers (types-only)
│   ├── free_prove.py         # WO-4: per-pair candidate collection, intersection, frozen pick
│   ├── transport.py          # WO-5: transport_types(), disjointify(), per-block template checks
│   ├── quotas.py             # WO-6: choose_Y0(), quotas(), palette reports
│   ├── fill.py               # WO-7: rank-fill meet, idempotence check
│   ├── solve.py              # WO-8: end-to-end batch solve; orchestrates modules above
│   └── audit.py              # WO-8: CLI to dump receipts for a task id
├── scripts/                  # thin CLIs (optional; can call arc.solve.main)
│   ├── solve.py
│   └── audit.py
├── tests/                    # WO-scoped tests; run on curated minis + invariants
│   ├── test_pi.py            # Π idempotence, stability, partition totals
│   ├── test_free_simple.py   # simple verifiers on microcases
│   ├── test_free_sbs.py      # SBS-Y / SBS-param proofs on microcases (incl. 007bbfb7)
│   ├── test_transport.py     # shape, block template hashes, disjointify no-bleed
│   ├── test_quotas.py        # ∑K==|S|, palette policy, no input minting
│   ├── test_fill.py          # quota satisfaction, rank minimality, idempotence
│   └── test_runner_smoke.py  # full-corpus smoke: receipts-only then solve subset
├── data/                     # (optional) tiny curated cases for tests
│   ├── mini_cases.json
│   └── 007bbfb7.json
├── outputs/                  # generated; ignored by git
│   ├── predictions.json      # id -> grid
│   ├── receipts.jsonl        # one JSON object per task, multi-WO receipts
│   └── summary.txt           # counts by FREE tuple; coverage
└── .gitignore
```

### Notes for WOs to reference

* **Module boundaries** map 1:1 to WOs:

  * WO-1 → `arc/pi.py`
  * WO-2 → `arc/receipts.py` (+ minimal harness plumbing in `arc/solve.py`)
  * WO-3A/3B → `arc/free_simple.py`
  * WO-3C/3D → `arc/free_sbs.py`
  * WO-4 → `arc/free_prove.py`
  * WO-5 → `arc/transport.py`
  * WO-6 → `arc/quotas.py`
  * WO-7 → `arc/fill.py`
  * WO-8 → `arc/solve.py`, `arc/audit.py`
* **CLIs**: Either use `scripts/solve.py` and `scripts/audit.py`, or expose `arc.solve:main()` and wire with `python -m arc.solve`.
* **Receipts**: all modules write via `arc/receipts.py` (single schema, single writer) so the reviewer has one JSONL to inspect.
* **Tests**: keep per-WO tests local to the module; add a smoke test for the runner.

If you want, I can now draft WO-1 (Π ruler) with exact function signatures, acceptance checks, and the receipts fields it must output.
