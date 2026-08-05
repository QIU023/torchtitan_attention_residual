# Matrix scripts — how the published numbers were produced

Everything in the RFC update and the eager-PR comment came out of these. They
were living in `/tmp` until now, which meant the numbers were not reproducible
by anyone but this box; that was a real gap, not a formality.

## Files

| file | what it does |
|---|---|
| `run13_twin_par.sh` | the 13 degree-2 cells; 2-GPU cells run concurrently on disjoint GPU sets, 8-GPU cells serial |
| `run13_flav.sh` | same, parameterised by `FLAVOR` and `ENTRY` |
| `run_maxdeg.sh` | the 5 max-degree cells (`ep8`, `pp4`, `pp8`, `tp4`, `cp4`) |
| `collect13.sh` | parses per-leg logs into a table; deliberately separate from the runners so a parsing bug never costs a re-run |
| `rerun_failed.sh` | re-runs short/missing legs on a fresh port, retrying ONLY on `EADDRINUSE` |
| `run_shim.py`, `gmm_shim.py` | entry point that installs the grouped_mm shim; used only for the clearly-labelled shimmed compile rows |
| `grouped_mm_repro.py` | the five-line reproduction of the upstream operator limitation |

Environment variables: `FLAVOR`, `OUT`, `STEPS`, `EXTRA` (e.g.
`--compile.enable`), `ENTRY` (e.g. `matrix_scripts/run_shim.py`).

## Environment the numbers were measured on

    torch      2.14.0.dev20260802+cu130   (CUDA 13.0)
    fla-core   0.5.1
    GPU        8 x NVIDIA GeForce RTX 5060 Ti, 15.5 GiB, sm_120
               dynamic shared memory per block (opt-in): 101376 bytes

**This hardware matters for three of the findings, and a maintainer on
datacenter GPUs will not reproduce them:**

* The KDA kernel asks for 108160 bytes of dynamic shared memory when its
  operands are fp32, above this card's 101376. On A100 (164 KB) or H100
  (227 KB) it fits, so the fp32 legs that failed here would pass there.
* `PP8xVP4` at the flavor's own `seq_len 8192` exceeds 15.5 GiB per GPU. The
  published run is at `seq_len 1024` for that reason and says so.
* Anything memory-bound generally. The loss numbers themselves are not
  hardware-dependent, but which cells run is.

## Pending re-verification on 8 x RTX 5090 / H200

Gated on hardware, not on code, and listed so nothing here reads as finished
that is not:

* `PP8xVP4` at `seq_len 8192` -- the configuration the pressure test was built
  for. The published run is 1024.
* Long-horizon training behind the 100-step smoke numbers. Those overfit a tiny
  dataset by construction and cannot speak to convergence.
* The shared-memory-bound cells, to confirm the 101376-byte attribution is the
  card and not something of ours that merely correlates with it.

## Two operational rules learned the hard way

* Wipe the per-leg dump folder before each run. A leftover checkpoint makes the
  trainer RESUME, so a 10-step run after a 3-step one produces 7 rows and the
  collector reads it as a failure. This cost three separate matrices.
* Never edit a source file while a matrix is running. Two runs were discarded
  for this; legs launched before and after the edit ran different code.

## Scope: multimodal only

The reported matrix runs the MULTIMODAL flavor. Text-only and dense legs exist as
controls for specific questions -- whether the run-horizon band is
architecture-specific, and whether a cross-layout disagreement is MoE route
flipping -- and should be run when those questions come up, not by default.
