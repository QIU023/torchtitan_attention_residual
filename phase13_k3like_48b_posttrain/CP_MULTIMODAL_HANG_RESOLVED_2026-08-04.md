# The CP multimodal hang: root cause, fix, and what it invalidates

Refs: pytorch/torchtitan#3029, pytorch/torchtitan#4025

Supersedes `HANDOFF_CP_HANG_2026-08-04.md`, whose final section attributed the
hang to gradient accumulation. That was the trigger, not the cause.

## The defect

Under CP a rank's sequence shard can legitimately hold **zero** vision
sentinels -- every image's tokens landed on its CP peer. The forward already
handled the collectives that fire on the way in: the tower still runs, so its
FSDP all-gathers match its peers'. But the branch then returned the text-only
result and **discarded the tower's output**:

```python
features = self._select_cp_shard(features, num_rows, cp_counts)   # -> 0 rows
if num_sentinels == 0:
    return self.language_model(input_ids)      # features dropped here
```

Dropping it takes the tower out of the loss graph. FSDP2 hangs a module's
reduce-scatter off the autograd hooks on that module's *output*, so this rank
issued one fewer gradient reduction than its peers, and the step deadlocked --
an NCCL watchdog timeout on `mesh_cp` and `mesh_fsdp` at once, not an error.

This is the same hazard `add_zero_valued_dependency` was added for on the
image-free (`pixel_values is None`) path. It was reached by a different route
and missed there.

A second, latent defect fell out of the fix: a non-last PP stage returns
`(hidden_state, block_residuals)` -- the AttnRes adapter ships the block
payload alongside -- and `add_zero_valued_dependency` takes a tensor. Both
tower-alive call sites now go through a local helper that puts the edge on the
hidden state and rebuilds the tuple, the same thing the adapter's own
`_keepalive_touch` does. The helper is deliberately kept out of
`add_zero_valued_dependency`, so that vendored copy stays byte-identical to
#4025's and the rebase remains a clean delete.

## How it was found

The Python stack of a hung rank is misleading: it parks on whatever CUDA sync
comes next, not on the collective that never matched. Two earlier probes
(per-step counts, faulthandler stacks) therefore could not localize it, and one
published a wrong conclusion.

What settled it was a per-rank trace of **every** collective -- `all_gather_single`,
`reduce_scatter_single`, `all_to_all_single`, `all_reduce` -- flushed and fsync'd
on entry so a partial trace survives the hang, with the model's own forward and
microbatch boundaries interleaved as markers. Diffing rank 4 against rank 6 (a
CP pair) gives a single divergence index:

```
    513  reduce_scatter_single   | reduce_scatter_single
    514  reduce_scatter_single   | reduce_scatter_single
    515  reduce_scatter_single   | MARK.mb.exit          <-- rank6 skips one
```

All 514 preceding entries identical, and rank 6 is the rank whose forward
marker reported `sent=0`. After the fix all eight ranks issue an identical
2065-entry sequence.

The instrument is `/tmp/cp_coll_probe.py`. It is worth keeping: for any
collective-desynchronization hang it converts "which rank is stuck where" from
inference into a diff.

## Correction to the previous attribution

`HANDOFF_CP_HANG_2026-08-04.md` concluded "the trigger is gradient
accumulation, and it hits k3mini too". The observation was right and the
attribution was wrong.

Gradient accumulation matters only because it shrinks the per-microbatch batch
to a single sequence, which is what lets one image land entirely in one CP
half and leave the other half with no sentinel. With `local_batch_size 4` the
four concatenated sequences essentially always put a sentinel in both halves,
so the branch is never taken. The defect is in the zero-sentinel shard.

That also explains the flavor asymmetry that looked mysterious: nothing about
`kimi_k3_debugmodel_pr_4025` versus `kimi_k3_mini_vl` was relevant. Both hang
with accumulation on, both pass with it off.

## Consequence for the published 12-leg multimodal matrix

The earlier 12/12 ran at `local_batch_size 4`, so it never exercised a
zero-sentinel shard. It was accurate for what it ran and **did not cover
gradient accumulation**, which is standard at any real scale.

Re-run after the fix, `kimi_k3_mini_vl` at dp2 x tp2 x cp2 with
`local_batch_size 1` (four microbatches), 10 steps:

    7.70459 7.65565 7.57014 7.38306 7.05885 6.85604 6.28894 5.74512 5.42987 5.31034

Previously this hung at step 2.

## Superseded below: the "hardware ceiling" was a config gap on our side

The section that follows recorded the three blocked legs as an unavoidable
limit of this GPU. That reading was incomplete, and the correction is in
`TWIN_FIDELITY_2026-08-04.md`: `training.dtype` was left at the float32 chain
default while #4025's debugmodel sets bfloat16, so the twin was running a
configuration that PR does not. Everything measured below is accurate for
what it ran; the conclusion drawn from it was too generous to us.

## Twin-flavor matrix after the fix: 10/13, and why not 13

`kimi_k3_debugmodel_pr_4025`, 3 steps, seed 42, deterministic, global batch 8:

| leg | step losses |
|---|---|
| fsdp2 | 12.05716 12.02725 12.00053 |
| cp2 | 12.05658 12.03887 12.00651 |
| fsdp2_tp2_pp2 | 12.07224 12.02164 11.96327 |
| fsdp2_tp2_cp2 | 12.04275 12.02269 11.98748 |
| tp2_pp2_cp2 | 12.07205 12.02185 11.98569 |
| fsdp2_pp2_cp2 | 12.05744 12.03855 11.97321 |
| ep2_fsdp2 | 12.05716 12.02780 12.00408 |
| ep2_fsdp2_tp2_pp2 | 12.05141 12.04240 11.96635 |
| ep2_fsdp2_tp2_cp2 | 12.06655 12.02728 12.00836 |
| ep2_fsdp2_pp2_cp2 | 12.06617 12.02984 11.96501 |
| **dp1** | blocked, see below |
| **pp2** | blocked |
| **tp2** | blocked |

Spread across the ten: 12.04275 to 12.07224 at step 1, i.e. 0.03 on a loss of
12, well inside the tolerance the other matrices use.

The three blocked legs fail identically:

    tvm.error.InternalError: Failed to set the allowed dynamic shared memory
    size to 108160

**This is a hardware limit of the box, not a defect, and not ours.** The GPU is
an RTX 5060 Ti (consumer Blackwell, cc 12.0) whose opt-in dynamic shared memory
per block is 101376 bytes. The KDA kernel at this flavor's head dimensions asks
for 108160 when its operands are fp32.

The pattern is exactly explained by parameter dtype, and the explanation is
falsifiable and was falsified in the right direction:

* All three blocked legs have `data_parallel_shard_degree 1` and no CP, so the
  FSDP mesh has size 1 and FSDP is not applied -- and `mixed_precision_param`
  is consumed only by FSDP, so parameters stay fp32.
* `cp2` also has `dp_shard=1` yet passes, because torchtitan's FSDP mesh is
  `dp_shard x cp`: turning CP on makes the mesh size 2, FSDP applies, params
  become bf16, and the kernel fits.
* **Prediction, then test:** forcing `--training.mixed_precision_param float32`
  on the passing `fsdp2` leg should reproduce it. It does, with the identical
  108160.
* Batch size is not involved: `local_batch_size` 1, 2 and 4 all request 108160.

So on any datacenter GPU (A100 164 KB, H100 227 KB opt-in) all thirteen run.
On this box, ten is the ceiling for this flavor in eager, and the three
unreachable legs are unreachable for a reason that has nothing to do with the
parallelism code. They were failing this way before these fixes too -- the
fixes moved the matrix from 6/13 to 10/13 and did not touch these three.

Stating the ceiling rather than quietly reporting 10/10.
