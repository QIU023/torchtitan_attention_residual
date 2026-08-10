# The 18-cell matrix with DEP and dynamic CP on, multimodal and text (2026-08-10)

Every earlier 18-cell table was multimodal only, and its three TP+CP cells were run with
`KIMI_VIT_DYNAMIC_CP=0` to get past a known gap. This run leaves both knobs on and adds a
text-only arm, so "is our 5D parallelism sound" can be separated from anything the vision
path contributes.

Baseline for comparison is `MATRIX_18_SDPA_2026-08-09.md`, parsed out of that file by
`matrix_scripts/compare_to_baseline.py` rather than retyped.

## Multimodal: `kimi_k3_debugmodel_report_arch`, DEP=1, dynamic CP=1, eager, 10 steps

15 cells runnable of 18, and the split is exact: **every cell with `pp>1` moved, every cell
with `pp=1` is bit-identical to the baseline.**

Bit-identical, all 10 steps (0.00e+00):

    dp1  fsdp2  cp2  tp2  ep2_fsdp2  ep8_fsdp8  tp4  cp4

Moved, and only these:

| cell | step-1 rel | step-10 rel | baseline -> new (step 1) |
|---|---|---|---|
| `fsdp2_tp2_pp2` | 7.5e-06 | 1.4e-03 | 12.04989 -> 12.04980 |
| `ep2_fsdp2_pp2_cp2` | 1.3e-04 | 5.5e-04 | 12.06966 -> 12.06804 |
| `ep2_fsdp2_tp2_pp2` | 2.7e-04 | 6.5e-04 | 12.05574 -> 12.05902 |
| `pp4` | 1.1e-03 | 2.8e-06 | 12.08310 -> 12.06958 |
| `fsdp2_pp2_cp2` | 2.1e-03 | 2.2e-03 | 12.02955 -> 12.05422 |
| `pp2` | 2.6e-03 | 1.0e-02 | 12.04691 -> 12.07827 |
| `pp8` | 2.6e-03 | 1.1e-02 | 12.03501 -> 12.06663 |

All ten runnable cells of the 13-cell half converge over the baseline's range
(~12.05 to 9.74-9.93 in ten steps), and the five max-degree cells match theirs.

**Why the PP cells move, with the control run rather than inferred.** The `pp2` cell was
re-run twice, cold, changing nothing but `KIMI_VIT_DEP`:

    DEP off   12.04691 11.97712 11.76462 11.38399 10.79868 10.41735 10.04876 9.91578 9.81886 9.80718
    baseline  12.04691 11.97712 11.76462 11.38399 10.79868 10.41735 10.04876 9.91578 9.81886 9.80718
    DEP on    12.07827 11.98252 11.79238 11.46894 10.98114 10.54952 10.17764 10.03643 9.94976 9.90757

DEP off reproduces the published baseline to every printed digit at all ten steps, and DEP
on reproduces this run's matrix cell. So DEP is the only variable and the shift is
deterministic, not run-to-run noise.

What the shift IS: DEP relocates the vision tower onto its own pipeline stage, so the stage
split changes; torchtitan seeds PP ranks distinctly, so a module that moves to another
stage is initialized from a different seed, and these cells are cold runs. That mechanism
was measured directly today -- `DEP_30L_RESOLVED_2026-08-10.md`, same 880 parameters,
global norm sum 7732.368 against 7733.152 before any forward -- and from a shared
checkpoint DEP off and DEP on are identical, including with the tower split across two
stages. A knob that only moves pipeline stages moving only pipeline cells, deterministically,
is consistent with that and with nothing else here.

### The three TP+CP cells fail, with the recorded cause

    NotImplementedError: Operator c10d.allgather_.default does not have a sharding
    strategy registered.

`fsdp2_tp2_cp2`, `tp2_pp2_cp2`, `ep2_fsdp2_tp2_cp2` -- the same three as the baseline, and
this is defect 2 exactly as documented in `MATRIX_18_CORRECTED_2026-08-09.md`:
`_attend_gather_kv` calls `dist_nn.all_gather` on `k` and `v`, which are DTensors under
vision TP. Fixing it means settling the local/DTensor contract through `wo`, which is
untouched. **This is the recorded defect showing when dynamic CP is left on, not a new
regression** -- the baseline's own three failures were failures with dynamic CP on too;
that table only reported them as passing when the knob was forced off.

## Text-only: `kimi_k3_mini_block_attn_res`, dynamic CP=1, eager, 10 steps

Chosen because it is the smallest text flavor that can express every cell -- `pp8` needs
8+ layers, `tp4`/`cp4` need 4+ heads, `ep8` needs 8 experts -- while still carrying KDA,
MoE and AttnRes rather than being a degenerate model. Run at `--training.seq_len 256
--training.dtype bfloat16` to match the multimodal arm's conditions; the flavor's own
defaults are seq 8192 / float32, which does not fit 16 GiB at this batch.

There is no published baseline for this flavor, so this table IS the baseline. What it
answers is convergence and expressibility per cell, not agreement with a prior number.

**18 of 18 cells pass.** No failures, no cells inexpressible.

```
dp1                    7.69827 7.69289 7.67392 7.60441 7.54067 7.39899 7.19878 6.85705 6.13021 5.35296
fsdp2                  7.69883 7.69801 7.70005 7.61676 7.53429 7.42179 7.24580 6.88431 6.38903 5.50735
pp2                    7.70316 7.69099 7.68393 7.62721 7.55041 7.42392 7.24637 6.92817 6.30139 5.53063
cp2                    7.70116 7.71964 7.68079 7.63141 7.54940 7.42569 7.23564 6.89572 6.28416 5.51992
tp2                    7.71995 7.72298 7.69589 7.65724 7.55978 7.43126 7.26407 6.93900 6.34618 5.58669
fsdp2_tp2_pp2          7.71848 7.69803 7.68536 7.60889 7.54259 7.43802 7.25529 6.91167 6.38249 5.50759
fsdp2_tp2_cp2          7.70269 7.68031 7.68590 7.63403 7.53508 7.40517 7.23179 6.88068 6.37915 5.49596
tp2_pp2_cp2            7.70745 7.72160 7.68244 7.63475 7.54461 7.41014 7.23647 6.90022 6.28760 5.50653
fsdp2_pp2_cp2          7.70943 7.71007 7.68412 7.62983 7.52767 7.42807 7.24636 6.89289 6.39772 5.51918
ep2_fsdp2              7.69883 7.70051 7.68976 7.63170 7.53801 7.41943 7.25781 6.88667 6.36743 5.50152
ep2_fsdp2_tp2_pp2      7.71173 7.71348 7.68440 7.65242 7.53315 7.41593 7.26947 6.94068 6.43372 5.57252
ep2_fsdp2_tp2_cp2      7.71307 7.70658 7.68946 7.66105 7.52567 7.41011 7.25053 6.89252 6.36766 5.48182
ep2_fsdp2_pp2_cp2      7.71081 7.70084 7.68805 7.62832 7.52949 7.43535 7.24035 6.91539 6.39642 5.51208
```

Max-degree:

```
ep8_fsdp8    step1 7.71147  step10 5.52316
pp4          step1 7.70411  step10 5.44017
pp8          step1 7.69957  step10 5.42214
tp4          step1 7.72012  step10 5.51731
cp4          step1 7.70856  step10 5.53288
```

Every cell starts at 7.698-7.720 and lands at 5.35-5.59 in ten steps. The spread across
eighteen different shardings of the same model is 0.24 at step 10, and 0.022 at step 1.

### The three TP+CP cells pass here, which locates defect 2

`fsdp2_tp2_cp2`, `tp2_pp2_cp2` and `ep2_fsdp2_tp2_cp2` are the exact three cells that fail
on the multimodal arm, and on text they run and converge like any other cell. So the
`c10d.allgather_.default` failure is in the **vision** TP x dynamic-CP path specifically,
not in TP+CP plumbing. That is now a two-arm comparison rather than a reading of the
traceback.

It also answers a question that was open from a different direction -- whether
full-parameter 5D parallelism itself was sound, independent of the vision path. On this
flavor, 18 of 18 shardings converge into a 0.24-wide band.

### DEP is not inert on a text model, it is invalid

`KIMI_VIT_DEP=1` on a text-only flavor fails every PP cell with

    RuntimeError: KIMI_VIT_DEP_STAGES=1: this rank owns 1 vision stage(s) by stage index
    but 0 were wired; an unwired share passes activations through unprocessed and
    reports no error

which is the finding-50 guard working: a rank is assigned a vision stage by index and
nothing can wire it, because there is no tower. So the text arm runs with DEP off. Dynamic
CP is genuinely inert there -- `KIMI_VIT_DYNAMIC_CP` is read only inside the multimodal
forward, which a text flavor never enters.

The guard's message is right about what it detected and unhelpful about the cause. A
text-only model with DEP requested should say so directly; that is a small follow-up, not
part of this run.

## Two things that cost a run, both mine

The text arm was launched once before this and every non-CP cell failed, for two reasons I
should have checked before launching rather than after:

* **DEP on a text model is invalid, not inert.** I assumed a model with no vision tower
  would ignore `KIMI_VIT_DEP`. It does not -- the guard above fires and every PP cell dies.
* **I never read the text flavor's own defaults.** `kimi_k3_mini_block_attn_res` is seq 8192
  / float32 against the multimodal flavor's seq 256 / bfloat16, so `dp1`, `fsdp2` and `tp2`
  OOM'd on 16 GiB cards. The corrected arm pins seq 256 / bfloat16 and one cell was probed
  for fit (1.04 GiB) before committing eight GPUs to eighteen.

Also worth recording because it reads exactly like a regression: **`collect13.sh` on a live
output directory reports in-flight cells as `FAIL (n/10)`**, and a partial step count is
indistinguishable from a crash. Two cells were misread that way here before the arm
finished. Collect once, after the runner prints DONE.

## Reading notes

`run_maxdeg.sh`'s console line prints `head -3` of the loss list without deduplicating
ranks, so a multi-rank cell prints step 1 three times -- `ep8_fsdp8 12.01898 12.01898
12.01898` reads like a frozen loss and is not one. Every number in this document comes
from the deduplicated per-step extraction, the same one `collect13.sh` uses.

## Repro

    OUT=/workspace/mx_dep STEPS=10 bash matrix_scripts/run_matrix_dep_dyncp.sh
    # the text arm needs EXTRA="--training.seq_len 256 --training.dtype bfloat16" and
    # KIMI_VIT_DEP unset; see the section above
    bash matrix_scripts/collect13.sh /workspace/mx_dep/mm_13 10 > /workspace/mx_dep/mm_13.txt
    python matrix_scripts/compare_to_baseline.py /workspace/mx_dep/mm_13.txt
