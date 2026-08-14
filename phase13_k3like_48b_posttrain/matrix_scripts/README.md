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

## Correctness checking (added 2026-08-07)

`replicate_axis_check.py` is the matrix's floor-free assertion, and the reason it
exists is that two real defects passed eighteen cells at 100 steps. Full reasoning,
the measured floor of the alternative, and the coverage table:
`../MATRIX_CORRECTNESS_2026-08-07.md`.

Short version: the matrix judges by LOSS AGREEMENT, whose band is 0.009-0.0146. The
LoRA `o_proj` defect was 2.9e-02 in step-1 loss and the MLA CP defect was 1-6% on
one parameter's gradient; both fit inside. More steps widen the band rather than
narrowing it, so the fix is a different KIND of judgement, not a longer run.

    PYTHONPATH=$TITAN torchrun --nproc_per_node=4 \
      <logbook>/phase13_k3like_48b_posttrain/matrix_scripts/replicate_axis_check.py \
      --module kimi_k3 --config kimi_k3_debugmodel_report_arch \
      --debug.seed 42 --debug.deterministic --training.steps 3 \
      --training.global-batch-size 8 --checkpoint.interval 100000 \
      --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2

**A collector must key on the `REPLICATE-CHECK` line, never on the exit code
alone.** An import error, an OOM and a port collision all exit 1, and a missing
PYTHONPATH once read as four cells finding problems. Three outcomes:

| line | meaning |
|---|---|
| `REPLICATE-CHECK PASS` | every replicated gradient bit-identical across its axis |
| `REPLICATE-CHECK FAIL: N ... differ` | real disagreement, world-reduced across ranks |
| `REPLICATE-CHECK NOASSERT` | this cell has no Replicate axis; not covered, not broken |
| line absent | the check never ran -- which is not the same as passing |

Coverage is TP-bearing cells. CP-only and FSDP-only cells have no Replicate axis
(the `"fsdp"` mesh here is `dp_shard x cp`, so parameters are sharded on the cp axis
too) and report NOASSERT; for those the assertion is the cross-run per-parameter
comparison, whose verdict must be "within the measured floor", not "zero". The
floor: changing only the gradient-accumulation order on one GPU with no parallelism
gives a norm-weighted max of 0.0975. `run_maxdeg.sh` / `run13_flav.sh` should carry
that control as a permanent cell.

## Disk hygiene (added 2026-08-07, after the box filled mid-matrix)

The failure was worse than a lost run: at 100% full the agent harness could not
write a command's output file either, so bash stopped working entirely and nothing
could be diagnosed or cleaned from inside the box. `/tmp` held **227 GB** of
checkpoints from matrices whose evidence is the ~200 KB log beside them.

`prune_run_artifacts.sh [--dry-run] [roots...]` deletes `checkpoint/`
subdirectories and JIT caches. **It keeps `tb/`**, and that distinction is the
whole point: measured on one 13-cell 100-step matrix, `checkpoint/` is 9.0 GB,
`structured_logs/` 159 MB and `tb/` 1.3 MB -- so checkpoint is 97% of the space,
while tb is the one thing that must survive, because stdout prints five significant
digits and CLAUDE.md is explicit that a loss comparison has to be read from the
TensorBoard record. Deleting whole dump folders, which the first version did, would
have destroyed the only full-precision copy of every published matrix.

Never touched: `*.log`, `tb/`, JSON/JSONL dumps, anything whose folder name
contains `seed` (regenerating one costs a run and breaks a matrix's shared-init
protocol), git trees, the vLLM source build, and exported HF bundles. Getting the
seed guard right took three dry runs -- `/seed_` missed a folder named plainly
`seed`, the prefix form then missed `qat_seed` / `nokda_seed` / `ds_seed`, and once
the rule moved to deleting `checkpoint/` the guard had to test the PARENT, since
every candidate's basename is then `checkpoint`. **Run `--dry-run` first.**

First run reclaimed **242 GB**, taking the disk from 84% to 20%.

`disk_watchdog.sh` runs it automatically. Installed as a supervisor service
(`/opt/supervisor-scripts/disk_watchdog.sh`, `disk_watchdog.conf`), checking every
60s, pruning below `WATCHDOG_MIN_FREE_GB` (default **40**). 40 rather than 10
because one 8-GPU 100-step cell writes ~700 MB of checkpoint and eighteen write
~12 GB: at 10 GB the writer can outrun the sweep between two checks. It logs what
it freed, and it warns loudly when a prune frees nothing -- a watchdog that runs,
frees nothing and stays quiet is indistinguishable from one that is not running.

**`/tmp` and `/workspace` are the same overlay on this box.** Cleaning `/workspace`
alone does nothing for a full `/tmp`; assuming they were separate filesystems cost
a round of pointless cleanup.

Both runners now `rm -rf "$OUT/$name/checkpoint"` as each cell finishes, rather
than after the matrix -- the disk fills DURING a matrix, not after it.


## How long the three-arm gate actually takes

**About 40 minutes** for all 54 cells (3 arms x 13-cell + 5 maxdeg), measured
2026-08-14 on this box: `run_postmerge_gate.sh` started its first smoke at
00:03:38 and finished at 00:43:55.

Recorded because "about 2.5 hours" was repeated several times in conversation
without anyone checking it against a run. It is not in any document -- it only
survived by being restated. Budget the real number when deciding whether a
change is worth a full gate: at 40 minutes the answer is almost always yes,
which is a different decision from the one 2.5 hours implies.
