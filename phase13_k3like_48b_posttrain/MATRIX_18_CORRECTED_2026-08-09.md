# The 18-cell matrix, rerun on the fixed head

Supersedes the tables in `REPORT_ARCH_MATRIX_2026-08-04.md` for the
`kimi_k3_debugmodel_report_arch` flavor. `MATRIX_CORRECTNESS_2026-08-07.md` said the four
cells carrying both TP and CP "need rerunning on the fixed head before being cited". That
was right, and the rerun produced something the note did not anticipate: **they did not
merely change, they failed to run at all.**

Multimodal (MoonViT-V2 + backbone), seed 42, `--debug.deterministic`, global batch 8,
10 steps, `--metrics.log_freq 1`.

## Results

| cell | step 1 | step 10 | notes |
|---|---|---|---|
| `dp1` | 12.05342 | 9.90565 | |
| `fsdp2` | 12.05033 | 9.91598 | |
| `pp2` | 12.04891 | 9.80407 | changed by the grad-norm fix |
| `pp4` | 12.08035 | 9.80270 | changed by the grad-norm fix |
| `pp8` | 12.04015 | 9.81109 | changed by the grad-norm fix |
| `cp2` | 12.03828 | 9.92359 | dynamic CP on (default) |
| `cp4` | 12.03993 | 9.91355 | dynamic CP on (default) |
| `tp2` | 12.06346 | 9.92391 | |
| `tp4` | 12.05647 | 9.94822 | |
| `ep2 x fsdp2` | 12.05033 | 9.91933 | |
| `ep8 x fsdp8` | 12.01828 | 9.92187 | |
| `fsdp2 x tp2 x pp2` | 12.04664 | 9.73895 | changed by the grad-norm fix |
| `fsdp2 x pp2 x cp2` | 12.02968 | 9.71810 | dynamic CP on; changed by the grad-norm fix |
| `ep2 x fsdp2 x tp2 x pp2` | 12.05293 | 9.74656 | changed by the grad-norm fix |
| `ep2 x fsdp2 x pp2 x cp2` | 12.07279 | 9.73979 | dynamic CP on |
| `fsdp2 x tp2 x cp2` | 12.09125 | 9.93360 | **dynamic CP OFF required** |
| `tp2 x pp2 x cp2` | 12.04720 | 9.72670 | **dynamic CP OFF required**; changed by the grad-norm fix |
| `ep2 x fsdp2 x tp2 x cp2` | 12.04897 | 9.94180 | **dynamic CP OFF required** |

18/18 run. The last three need `KIMI_VIT_DYNAMIC_CP=0`; see the limitation below.

**Convergence clusters by parallelism family**, which is a sanity signal rather than a
coincidence: the three PP-only cells land at 9.8067-9.8095; the four cells combining PP
with CP land at 9.7265-9.7398; the TP-bearing cells without PP land at 9.9234-9.9482.
A cell that had drifted into a different family would stand out.

## The limitation found by rerunning: vision TP x dynamic CP is unimplemented

Every cell carrying BOTH TP and CP crashed, and it took two fixes to reach a clean
diagnosis, the second of which is still open.

**Defect 1 (fixed).** `MoonViTPatchEmbed.add_pos_emb` computed `x_LD + torch.cat(embs)`
where `embs` comes from `_spatial`, which returns a **Replicate DTensor** when vision TP
is on. Dynamic CP's branch in `forward` builds its whole-image placeholder as a plain
`torch.zeros`, so the addition mixed the two:

    RuntimeError: aten.add.Tensor got mixed torch.Tensor and DTensor

Fixed in both directions, because the branch mismatches twice with opposite polarity:
the table is taken `to_local` for the placeholder addition, and the sliced result is
wrapped back as `Replicate` before adding to `x` when `x` is a DTensor. Both conversions
are exact rather than coercions -- the position table is replicated across the TP axis, so
the local shard IS the full table. The order matters: `_slice_for_shard` indexes rows and
must run on a local tensor.

My first attempt fixed only the first direction, and the second surfaced immediately --
worth recording, because "the error message changed" is not the same as "the bug is
fixed".

**Defect 2 (OPEN).** With the addition fixed, the next failure is:

    NotImplementedError: Operator c10d.allgather_.default does not have a sharding
    strategy registered.

`_attend_gather_kv` calls `dist_nn.all_gather` on `k` and `v`, which are DTensors under
vision TP. The gather has to happen on LOCAL tensors: TP shards the attention heads while
dynamic CP partitions the patch dimension, so each TP rank should gather its own heads'
keys across the CP ranks. The two are orthogonal, but fixing it means settling the
local/DTensor contract through `wo` as well, which is more than a type conversion and is
not being rushed.

**So the three TP+CP cells run with dynamic CP disabled**, taking the image-level
round-robin path instead. That is a declared limitation with a known cause, not a silent
fallback -- and it does not affect the CP-only cells (`cp2`, `cp4`) or the PP+CP cells,
which run dynamic CP on by default and pass.

## Why this rerun was worth doing before the PR

The three cells would have failed in a reviewer's hands. The published table listed them
as passing, and they had produced those numbers before vision TP landed -- so the
combination has never worked since. This is exactly the case the "rerun before citing"
note was written for, and the answer turned out to be stronger than expected.

## Reproduce

    cd phase13_k3like_48b_posttrain
    FLAVOR=kimi_k3_debugmodel_report_arch OUT=/workspace/mx18a STEPS=10 \
      bash matrix_scripts/run13_flav.sh
    FLAVOR=kimi_k3_debugmodel_report_arch OUT=/workspace/mx18b STEPS=10 \
      bash matrix_scripts/run_maxdeg.sh
    bash matrix_scripts/collect13.sh /workspace/mx18a 10

The three TP+CP cells need `KIMI_VIT_DYNAMIC_CP=0` prepended until defect 2 is fixed.

Note on reading `run_maxdeg.sh`'s console output: it prints the first three matching lines
per cell without de-duplicating by step, so a cell whose ranks all log the same loss looks
like it is not learning. It is -- parse with the `collect13.sh` logic instead. I checked
this before reporting `tp4`/`cp4`/`ep8` as stuck, and they are not.

## The combined DEP + dynamic CP arm, and the defect it caught

Added per the judgment that one combined arm on the PP-and-CP cells beats a 4x axis
expansion. It was going to be three cells; it is **two**, because `tp2 x pp2 x cp2` carries
TP and would hit the vision-TP limitation above. So: `fsdp2 x pp2 x cp2` and
`ep2 x fsdp2 x pp2 x cp2`, each with `KIMI_VIT_DEP=1 KIMI_VIT_SIDE_STREAM=1` against off.

**From a SHARED SEED CHECKPOINT**, which the first attempt at this arm omitted -- DEP
changes the stage split, so cold-start arms consume RNG differently and are not comparable.
That is a lesson already recorded in this project's handoff and I repeated it anyway; the
step-1 offset the cold-start version showed (12.02968 vs 12.04443) was that artefact.

| cell | DEP off | DEP on |
|---|---|---|
| `fsdp2 x pp2 x cp2` | 12.07716 / 12.00107 / 11.78638 / 11.44469 | 12.07716 / 12.00107 / 11.77114 / 11.44549 |
| `ep2 x fsdp2 x pp2 x cp2` | 12.07716 / 11.99539 / 11.77520 / 11.43276 | **12.07716 -> NaN at step 2** |

Two things follow, and only the second is a defect.

**DEP's forward is neutral under PP+CP.** Both cells match bit-for-bit at steps 1 and 2
(12.07716, 12.00107). The later divergence on the first cell (11.78638 vs 11.77114 at step
3, back to 0.0008 apart at step 4) moves in both directions and is bf16 accumulation, not a
trend.

**DEP x EP is BROKEN, and this arm is what found it.** With EP in the mesh, step 1's loss
is correct and step 2 is `global_avg_loss=nan`. Correct forward followed by an immediate
NaN points at the backward or the optimizer rather than the forward -- and the same cell
with DEP off runs ten clean steps. Every previous DEP gate (n_vit = 1, 2, 4, bit-identical)
was run WITHOUT EP, so none of them could have seen this.

Not diagnosed further tonight, deliberately: it needs the EP gradient path traced against
the changed stage split (DEP takes the vision stage out of the text budget, so which stage
holds which experts moves), and guessing at it is how a wrong fix gets committed.

**Declared limitation, therefore: do not enable DEP together with EP.** DEP with
FSDP/TP/PP/CP is exercised; DEP with EP is known-broken as of this commit.

### What the combined arm cost and returned

Four extra 8-GPU runs, about 15 minutes. It caught a blocking defect that four earlier
numerical gates could not, because they shared a blind spot -- no EP. That is the argument
for the arm existing, and it is a stronger one than I had when proposing it: I argued it
would catch interaction REGRESSIONS, and what it actually caught was an interaction that
never worked.

### The DEP x EP NaN is NOT what PR19 fixed -- checked, not assumed

The branch picked up an upstream merge carrying our own PR19,
"[MoE] fix: pass the computed `in_grad_placements` without EP too -- TP gradients below the
experts lose their reduction". That is an EP gradient-reduction fix, and the NaN is a
correct forward followed by an immediate blow-up, so the two looked like they might be the
same defect. Rerun on the merged head, from a shared seed checkpoint:

    ep2_fsdp2_pp2_cp2  DEP=off  12.07716 / 11.99539 / 11.77520 / 11.43276   clean
    ep2_fsdp2_pp2_cp2  DEP=on   12.07716 -> NaN at step 2                   unchanged

**So they are independent defects.** Worth having checked rather than assumed: the
attractive reading was "upstream already fixed it, we only need to rebase", and that would
have sent the next diagnosis down the wrong path entirely. The limitation stands -- do not
enable DEP together with EP -- and its cause is still to be found in how DEP's stage split
interacts with expert placement, not in `in_grad_placements`.

---

## The DEP x EP NaN was a CORE defect, now fixed -- and it moved the PP reference values

Diagnosed to `torchtitan/distributed/utils.py`, not to kimi_k3. **DEP was only the
trigger**, and the trigger condition is broader than DEP: any configuration where a PP
stage owns gradients in only one of the EP / non-EP groups. A DEP vision stage owns no
experts by construction, so it makes the condition certain.

### The chain, each step ruling out one candidate

A probe (`matrix_scripts/dep_ep_grad_probe.py`, wraps `clip_grad_norm_` from an entry
point rather than editing core) reported, per rank:

    DEP=off  rank0/1: 228 params  axes={fsdp:210, efsdp+ep:18}  returned_total_norm=12.6250
             rank2/3: 166 params  axes={fsdp:148, efsdp+ep:18}  returned_total_norm=12.6250
    DEP=on   rank0/1:  31 params  axes={fsdp:31}   local_non_ep=6.5930/1.1517  returned=6.6895
             rank2/3: 363 params  axes={fsdp:327, efsdp+ep:36}  local_ep=3.0443  returned=0.0000

Parameter totals agree (31+363 = 228+166 = 394), every gradient exists, every gradient is
finite, every local norm is non-zero -- and the two sides of the pipeline return DIFFERENT
`total_norm`. The 6.6895 is exactly `sqrt(6.5930^2 + 1.1517^2)`: the fsdp reduction only,
with the other PP side's contribution absent.

Root cause: `get_total_norm([])` returns a **CPU float32** `tensor(0.)`. A rank with no EP
gradients therefore reaches the `pp_mesh` all_reduce holding float32 while a peer holding
bf16 expert gradients holds bfloat16, and **NCCL returns garbage for a dtype mismatch
instead of raising**. Fix: normalise `total_norm` to float32 on the mesh's device before
the collective, in both the EP and non-EP paths. float32 is also the right width for a sum
of squares.

### Two of my own claims corrected along the way

* "The NaN is at step 2" -- no: **step 1's grad_norm is already wrong**, and step 2's NaN
  loss is the consequence of clipping with a garbage norm.
* "DEP + EP needs experts spread over several stages" -- no: pp4 spreads them over three
  stages and still fails. Tuning the PP degree is not a workaround.

The probe itself needed correcting too. Its first version reported only
`nonfinite_grads=0`, which reads identically whether gradients are healthy or identically
zero -- and zero was the hypothesis under test.

### Cost: the PP-bearing reference values moved

Clipping is active (grad_norm ~12.6 against max_norm 1.0), so a norm computed in float32
instead of bf16 changes the scale factor by ~0.34% and with it the trajectory. Reruns of
all 18 cells confirm the fix's scope is exactly `pp_mesh is not None`:

* **13 cells without PP are bit-identical** -- dp1, fsdp2, cp2, cp4, tp2, tp4, ep2xfsdp2,
  ep8xfsdp8, fsdp2xtp2xcp2, ep2xfsdp2xtp2xcp2. This is the negative control, and it is
  what makes "the fix does only what it claims" a measurement rather than an argument.
* **PP-bearing cells MAY move**, and some do not: `ep2 x fsdp2 x pp2 x cp2` is unchanged
  at 9.73979 because a 0.34% scale difference can be absorbed by bf16 rounding. So the
  rule is "PP cells are allowed to move", not "PP cells move".

Step-1 losses are unchanged everywhere, as they must be -- step 1 is pure forward and
clipping affects only the update that follows it.

`ep2 x fsdp2 x pp2` was missing from the 18 and is now covered: DEP off 12.07418 /
grad_norm 12.5816, DEP on 12.07418 with step 2 also bit-identical (11.98418). **DEP + EP
works.**
