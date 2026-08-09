# Aligning DEP on with DEP off: it was never DEP

The ask was to make `KIMI_VIT_DEP=1` match `KIMI_VIT_DEP=0`, on the suspicion that the
overnight batch introduced a regression. It did not, and the divergence is not in DEP.

## What the traceback found

Same two arms (`ep2 x fsdp2 x pp2`, seed 42, deterministic, 3 steps), same flags:

| commit | DEP off step 1 | DEP on step 1 |
|---|---|---|
| `42b846b84` (2026-08-08 05:06, the commit that introduced DEP) | 12.07152 | 12.04214 |
| `3e7e23e3c` (overnight start) | 12.07152 | 12.04214 |
| `a6f582b3b` (after the overnight batch) | 12.07152 | 12.04214 |

Byte-identical across all three. **No regression from the overnight work, and none from
anything after DEP landed** -- the on/off gap has been there since DEP's first commit.

(Worth noting what the earliest commit also shows: DEP on reported `grad_norm 0.0000` and
`wiring=0` there. That was the real defect at the time, and the later wiring work fixed it;
the current head reports 13.4154.)

## Why the arms differed at all, and why that is correct

torchtitan seeds with `distinct_seed_mesh_dims=["pp"]` -- same seed within an SPMD group,
**different seeds across PP groups**. DEP takes a stage for the vision tower out of the text
budget, so with `pp=2`:

* DEP off: stage 0 = text layers 0-6 (PP seed A), stage 1 = text layers 7-12 (PP seed B)
* DEP on: stage 0 = the tower (PP seed A), stage 1 = **all** text layers (PP seed B)

Text layers 0-6 move from seed A to seed B, so the two arms start from different weights.
Both are valid initializations. Total parameters agree exactly (120,514,724), so nothing is
dropped. Bit-identity was never an achievable pass condition for on-vs-off, which is now
corrected in `run_dep_cp_cells.sh` and in MATRIX_18_CORRECTED.

## From a shared seed checkpoint: the forward and the gradients are exact

Removing initialization as a variable -- both arms load one checkpoint via
`--checkpoint.initial_load_path`:

| | DEP off | DEP on |
|---|---|---|
| step 1 loss | **11.81748** | **11.81748** |
| step 1 grad_norm | 10.008054 | 9.951641 |
| step 2 loss | 11.77884 | 11.77901 |

Loss identical to every printed digit. Then `dep_grad_multiset_probe.py` compared the
per-parameter gradient norms (read BEFORE the clip, which rescales in place -- the first
version of the probe read them after and the numbers were meaningless):

    788 shard-norm entries per arm, paired by shape: 788/788 identical, max relative
    difference 0.000e+00

**The gradients are exact.** So is the true norm: summing those 788 shard norms in float32
gives **9.989287 in both arms**.

## The actual defect is in core, and both reported norms are wrong

`torch.nn.utils.get_total_norm` returns the norm in the **gradients' dtype**. With bf16
gradients that is a bf16 scalar -- 3 to 4 significant digits -- so the reported value is
accurate to a few tenths of a percent, and the error depends on how the parameters are
grouped. Reproduced standalone on 394 synthetic bf16 tensors:

    float32 exact:              121.222923
    get_total_norm (bf16):      121.000000    0.184% error, dtype=torch.bfloat16
    split 100/294:              121.153351    0.057%
    split 200/194:              121.050613    0.142%
    split 300/94:               121.011543    0.174%

Partition-dependent, at exactly the observed magnitude. Against the true 9.989287:

| arm | reported | error |
|---|---|---|
| DEP off | 10.008054 | +0.19% |
| DEP on | 9.951641 | -0.38% |

And it is not cosmetic. `max_norm = 1.0` against a norm near 10 means **clipping fires every
step**, so the clip factor carries the error straight into the update: 0.099920 vs 0.100486,
a **0.566% difference in effective step size**. That is what makes step 2 diverge, and it is
the whole of the on-vs-off gap.

So: `clip_grad_norm_` reports and applies a grad norm whose error depends on the PP/EP
partition, for any bf16 run. Ours is not the only model affected -- llama3, qwen3,
deepseek_v3 and gpt_oss all take the same path.

## The fix, and why it is a decision rather than a commit

Compute the norm in float32. The shape of it: `get_total_norm` takes no dtype, so the
per-tensor norms have to be taken with `torch.linalg.vector_norm(g, norm_type,
dtype=torch.float32)` and combined in float32, in both `clip_grad_norm_` and
`_clip_grad_norm_with_ep`, keeping the existing DTensor `full_tensor()` handling.

Not landed here, for one reason: it changes the reported grad_norm, and therefore the
clipping factor, and therefore the loss trajectory, for **every bf16 model in torchtitan** --
the battle-tested paths CLAUDE.md says to protect and flag rather than quietly change. It is
also a clean upstream bug report on its own, with the standalone reproduction above and no
Kimi code in it.

Two ways to take it:

* **Fix it in core** and accept that the matrix references move (a correctness improvement:
  the new numbers are the accurate ones). This is what actually aligns DEP on with off,
  because the remaining gap is entirely this error.
* **Leave it and change the criterion.** DEP on and off already agree on everything DEP
  controls -- same forward, same gradients. Comparing their reported grad_norms is comparing
  two differently-wrong measurements of the same quantity.

My recommendation is the first, as a separate upstream PR paired with the grad-norm dtype fix
already sitting in `torchtitan/distributed/utils.py` (same function, same family of defect:
one was a dtype mismatch across `pp_mesh` that NCCL turned into garbage, this one is a
precision loss that the partition makes visible).

## Reproducing

    # seed checkpoint
    torchrun --nproc_per_node=4 -m torchtitan.train --module kimi_k3 \
      --config kimi_k3_debugmodel_report_arch --debug.seed 42 --debug.deterministic \
      --training.steps 1 --training.global-batch-size 8 --training.local-batch-size 2 \
      --parallelism.expert_parallel_degree 2 --parallelism.data_parallel_shard_degree 2 \
      --parallelism.pipeline_parallel_degree 2 --checkpoint.enable \
      --checkpoint.interval 1 --dump-folder /tmp/seedck

    # both arms, then compare
    for mode in off on; do
      env $( [ $mode = on ] && echo KIMI_VIT_DEP=1 ) torchrun --nproc_per_node=4 \
        matrix_scripts/dep_grad_multiset_probe.py --module kimi_k3 \
        --config kimi_k3_debugmodel_report_arch --debug.seed 42 --debug.deterministic \
        --training.steps 1 --training.global-batch-size 8 --training.local-batch-size 2 \
        --parallelism.expert_parallel_degree 2 --parallelism.data_parallel_shard_degree 2 \
        --parallelism.pipeline_parallel_degree 2 \
        --checkpoint.initial_load_path /tmp/seedck/checkpoint/step-1 \
        --dump-folder /tmp/gm_$mode > /tmp/gm_$mode.log 2>&1
    done
    python matrix_scripts/dep_grad_multiset_probe.py --compare /tmp/gm_off.log /tmp/gm_on.log
