# Handoff: the CP+TP hang on the PR-4025 twin flavor

Refs: pytorch/torchtitan#3029, pytorch/torchtitan#4025

State as of the end of this session. Written so the next one can resume without
re-deriving anything.

## The failure

    torchrun --nproc_per_node=8 -m torchtitan.train --module kimi_k3 \
      --config kimi_k3_debugmodel_pr_4025 --training.seq_len 512 \
      --debug.seed 42 --debug.deterministic --training.steps 5 \
      --training.global-batch-size 8 \
      --parallelism.data_parallel_shard_degree 2 \
      --parallelism.tensor_parallel_degree 2 \
      --parallelism.context_parallel_degree 2

Step 1 completes (loss 12.04275). Step 2 hangs. After 100 s the NCCL watchdog
fires on two process groups at once:

    [PG mesh_cp]   ALLREDUCE, NumelIn=2, NumelOut=2
    [PG mesh_fsdp] _ALLGATHER_BASE, NumelIn=10486144, NumelOut=41944576

`ep2_fsdp2_tp2_cp2` fails identically. Everything else on this flavor passes.

## Ruled out by measurement, not by argument

| hypothesis | how it was killed |
|---|---|
| KCP (fla's KDA context parallelism) | The timing-out collective is our own 2-element all_reduce in `_select_cp_shard`, not anything fla issues |
| The sentinel-count assertion fires on one rank | It never appears in any rank's log |
| A rank takes the wrong slice | rank 0 and rank 2 are a CP pair reporting 255 and 34, summing to 289 = 17x17 -- exactly one 34x34-patch image after 2x2 merge. The arithmetic is right |
| Ranks enter the collective a different number of times | Serialized per-rank dumps: all eight ranks at `forward=4, exchange=4` for step 1. Identical |

The fourth was published as a finding in commit 6ce9a1c and retracted in
80b62d6 -- it came from reading interleaved concurrent stdout, which is
inference, not measurement.

## Fixed along the way (both verified, both real)

* `kimi_k3_debugmodel_pr_4025` inherited k3mini's 15-entry `kda_layers` into a
  13-layer model. Deriving it from `full_attn_layers` repaired `ep2_fsdp2`.
* Two collectives were entered on a data-dependent condition
  (`pixel_values is None`): the sentinel-count exchange, and -- now that the
  tower is FSDP-sharded -- the tower's own pre-forward all-gather. Both now key
  off the mesh instead, and the image-free path runs the tower on a placeholder
  through `add_zero_valued_dependency`. That second one is exactly the hazard
  #4025 added that helper for, reached by a different route.

Neither was sufficient.

## What the evidence now says

Step 1 is fully aligned across all eight ranks. The divergence begins inside
step 2. Only step 1 appears in the dumps because the probe writes at the end of
`train_step` and step 2 never returns.

## Next probe, precisely

`/tmp/cp_count_probe.py` works but flushes per step. **Change it to flush on
every collective entry** so a partial step-2 trace survives the hang:

```python
def ex(self, local):
    LOG["rows"].append((LOG["step"], "exchange", int(local)))
    _flush()          # <- add this
    return _ex(self, local)
```

Run from `/workspace/torchtitan_attention_residual/torchtitan` with
`PYTHONPATH=$PWD` (running it from the parent directory fails with
`ModuleNotFoundError: No module named 'torchtitan.models'`).

Then read which rank stops adding rows first, and what the last row it wrote
was. That is a fact, not a hypothesis.

## The open question that matters more

**Why does `kimi_k3_mini_vl` not hit this?** Its 12-leg matrix passed 10 steps
including four CP legs (`mm_fsdp2_cp2`, `mm_fsdp2_pp2_cp2`, `mm_fsdp2_tp2_cp2`,
`mm_ep2_fsdp2_tp2_cp2`, `mm_ep2_fsdp2_pp2_cp2`) on the same code paths.

Two readings, and they have very different consequences:

1. **Config-driven.** The flavors differ in ways that could matter: vocab 2020
   vs 163840, dim 512 vs 256, 21 layers vs 13, and -- probably most relevant --
   `max_patches` 1024 vs unset, so the twin's images produce ~289 sentinels
   each against a 512-token sequence, i.e. one image nearly fills it and
   straddles the cp2 boundary. k3mini's smaller images may simply never produce
   the distribution that triggers this.

2. **The 12/12 was luck.** Same code, same collectives; if the trigger is a
   particular sentinel distribution across the shard boundary, k3mini_vl could
   hang on a different data shard or at step 50.

Tested. `kimi_k3_mini_vl` at dp2 x tp2 x cp2 runs **30 steps** clean --
7.73550 -> 2.78104, three times the horizon of the published 12/12, no hang.

So reading 2 (luck) is not supported: the twin fails at step 2 while k3mini
survives 30, which is a difference in configuration rather than in chance. The
12/12 stands at 30 steps for the leg most likely to be fragile.

## SOLVED: the trigger is gradient accumulation, not the model

Neither flavor is special. The published multimodal matrix passed
`--training.local-batch-size 4`; the twin matrix did not, and both flavors
default to 1. With global batch 8 over dp2, local batch 4 means one forward per
step and local batch 1 means **four gradient-accumulation microbatches** --
which is exactly the `forward=4` the probe recorded.

| flavor | local_batch 4 (no accumulation) | local_batch 1 (4 microbatches) |
|---|---|---|
| `kimi_k3_debugmodel_pr_4025` | 5 steps pass | hangs at step 2 |
| `kimi_k3_mini_vl` | 30 steps pass | **hangs at step 2** |

So the defect reproduces on k3mini_vl too. It is a real defect in the CP
multimodal path under gradient accumulation, and the earlier 12/12 never
exercised it because that run happened to pass a local batch large enough to
avoid accumulation entirely.

**Consequence for the published result: the 12-leg multimodal matrix is
"passes without gradient accumulation", not "passes". That qualification has to
travel with the number**, since accumulation is standard at any real scale.

Why accumulation breaks it is not yet established. The shard arithmetic is
right and the per-step call counts match at step 1, so the suspect is state
that persists across microbatches within a step -- the CP count exchange runs
once per microbatch, and something about the second step's first microbatch
diverges.

## Superseded: which configuration difference triggers it
Both flavors use `max_patches=1024` and `seq_len 512`, so the obvious candidate
is ruled out. Remaining differences: vocab 2020 vs 163840, dim 512 vs 256,
21 layers vs 13, 15 KDA layers vs 10, and `local_batch_size` 4 vs unset. The
next step is to bisect these by mutating the twin one field at a time toward
k3mini until the hang disappears -- each run is ~2 minutes, so this is cheap
and mechanical.

## Files

* Flavor: `torchtitan/models/kimi_k3/config_registry.py::kimi_k3_debugmodel_pr_4025`
* Code under suspicion: `torchtitan/models/kimi_k3/multimodal_model.py`,
  `_exchange_sentinel_counts` / `_select_cp_shard` / `forward`
* Probes: `/tmp/cp_count_probe.py` (per-rank counts),
  `phase13_k3like_48b_posttrain/vision_liveness_probe.py`,
  `phase13_k3like_48b_posttrain/replicated_grad_rank_probe.py`,
  `phase13_k3like_48b_posttrain/grad_attrib_probe.py`
* #4025 worktree: `/tmp/wt_4025` (branch `pr4025`, upstream's unmodified
  `models/common/moe.py`, useful as a control tree)
