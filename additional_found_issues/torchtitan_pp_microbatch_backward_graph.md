# Issue 1: torchtitan PP + Interleaved1F1B + LOCAL_BS ≥ 3 + V ≥ 2 → "Trying to backward through the graph a second time"

> **This is a software BUG, not a hardware bottleneck.** The error fires
> inside the autograd graph traversal logic in pytorch's `pipelining/_backward.py`,
> independent of which interconnect (PCIe / NVLink / IB) is used. The
> repro would crash identically on an H100 NVLink box. Distinguish from
> the PCIe-bandwidth observation in `phase7/THROUGHPUT_BOTTLENECK_ANALYSIS.md`,
> which is a hardware trade-off (renting RTX 5090 PCIe instead of an
> NVLink box) — that is not a bug.



## Symptom

When using torchtitan's pipeline-parallel path with:
- `--parallelism.pipeline_parallel_schedule Interleaved1F1B`
- `--parallelism.pipeline_parallel_layers_per_stage V` with **V ≥ 2** (virtual stages)
- `--training.local_batch_size LBS` with **LBS ≥ 3**

every torchrun rank crashes during the **second** training step's backward
with:

```
RuntimeError: Trying to backward through the graph a second time
(or directly access saved tensors after they have already been freed).
Saved intermediate values of the graph are freed when you call
.backward() or autograd.grad(). Specify retain_graph=True if you need
to backward through the graph a second time or if you need to access
saved tensors after calling backward.
```

Stack trace points to
`/venv/main/lib/python3.14/site-packages/torch/distributed/pipelining/_backward.py:411` →
`stage_backward(...)` reusing autograd-saved tensors that were already
freed by an earlier microbatch's `.backward()` call.

## Reproductions on this 8-GPU PCIe box

| Config | Mesh | LBS | GBS | V | Outcome |
|---|---|---|---|---|---|
| A2 alignment | FSDP=2 × PP=4 | 1 | 16 | 2 | ✅ 500 steps clean |
| A3 alignment | FSDP=2 × PP=2 × TP=2 | 1 | 16 | 2 | ✅ 500 steps clean |
| A2 tier_b | FSDP=2 × PP=4 | **5** | 120 | 2 | ❌ crash on step 2 backward |
| A2 tier_a | FSDP=2 × PP=4 | **8** | 384 | 2 | ❌ crash on step 2 backward |
| v10 (1st attempt) | FSDP=2 × PP=2 × TP=2 | **15** | 120 | 2 | ❌ schedule rejected (rounds=7 mismatch) before backward, but if it had passed it would crash for the same reason |
| v10 (2nd attempt) | FSDP=2 × PP=2 × TP=2 | **14** | 112 | 2 | ❌ crash on step 2 backward |
| v10 (3rd attempt) | FSDP=2 × PP=2 × TP=2 | **1** | 14 | 2 | ✅ trains stably (current run) |
| B0 baseline | FSDP=8 (no PP) | 2 | 16 | 1 | ✅ — no PP path, no microbatched backward |

**Pattern:** PP backward graph reuse fires only when **all three** conditions hold:
- pipeline_parallel_degree > 1 (PP active)
- pipeline_parallel_layers_per_stage > 1 (V > 1, virtual stages enabled)
- local_batch_size ≥ 3 (each PP-microbatch is itself a non-trivial batch)

LBS=1 with the same V=2 PP=4 / PP=2 TP=2 mesh works (A2/A3 alignment proof).
Pure FSDP (B0) and FSDP+PP at V=1 (untested but theoretically) bypass the
microbatched backward path that reuses autograd state.

## Hypothesized root cause (preliminary)

The Interleaved1F1B schedule with V=2 issues `forward_microbatch_i`,
`backward_microbatch_j` actions in interleaved order across V virtual
stages. The autograd graph for microbatch `i` is built during `forward`
and intended to be consumed exactly once during the matching `backward`.

When LBS=1, each microbatch carries one example; the graph is small
and the schedule's interleaving works.

When LBS ≥ 3, each microbatch carries multiple examples. The fla-core
`chunk_kda` triton kernel processes all LBS examples in a single call
and writes a `state` tensor used by both the next forward microbatch
**and** the current backward microbatch. The autograd engine sees the
state tensor referenced by two different graph nodes; when the first
backward runs, it frees the saved tensors; the second backward (across
the V virtual stage boundary, still part of the same step) tries to
recompute and fails.

Alternative (less likely): the AttnRes `block_attn_res` aggregator
inside `KimiAttnResDecoderLayer.forward` keeps a list reference into
`partial_block` across microbatches in a way that aliasses with the
PP schedule's autograd save_for_backward set.

Both hypotheses are consistent with V=1 working (no virtual stage
boundary that interleaves backward inside a step) and LBS=1 working
(no multi-example state to alias).

## Why this is a torchtitan core issue, not our experiment's

The error fires inside
`torch/distributed/pipelining/_backward.py:411` (vendored pytorch
distributed code), reached via
`torchtitan/torchtitan/distributed/pipeline_parallel.py` from the
torchtitan trainer. The `experiments/kimi_linear/` code never touches
the autograd graph directly. We've reproduced the failure with all
three of our experiment-specific patches reverted (A3 fix, AttnRes
to_local, etc.) — the bug exists in the upstream PP + V≥2 path.

## Workarounds in this repo

1. **Force LBS=1 for any 3D PP+TP run** — used by v10 and all alignment
   runs. Costs throughput (need grad_accumulation to reach production
   batch sizes, and grad_accum step time scales linearly).
2. **Use 1F1B schedule (V=1) instead of Interleaved1F1B (V=2)** — gives
   up the V=2 throughput improvement but bypasses the bug. Not yet
   tested in this repo for kimi_linear; would need a separate
   alignment smoke.
3. **Disable PP entirely** — go to FSDP-only on the 8-GPU box for
   production-batch pretrain (this is what v10 was originally going
   to do before the user pinned 3D-must).

## Severity

- **High for production training** that wants per-rank LBS large enough
  to amortize compute across the PP bubble, on hardware with limited
  per-GPU memory.
- **Medium for cluster-replay traffic recording** (this project's
  primary deliverable) because LBS=1 still produces the correct NCCL
  pattern shape — only tensor sizes differ from production tier A.

## Candidate fixes (for upstream RFC)

1. Audit `_backward.py:stage_backward` for retained tensor sets across
   virtual-stage boundaries; clear or `.detach()` saved tensors that
   are reused across microbatches.
2. Add `retain_graph=True` to the backward call when V > 1 and
   detect-and-clear at the start of each top-level step.
3. Document the LBS×V interaction as a known limitation and reject
   LBS ≥ 3 + V ≥ 2 with a clear error at config-validation time, until
   the underlying autograd-graph-reuse bug is fixed.

## Reproduction recipe (minimal)

```bash
# In any torchtitan checkout with a kimi_linear-style flavor + Interleaved1F1B:
torchrun --nproc_per_node=8 -m phase5.train_mm \
  --module kimi_linear --config kimi_linear_436m_block_attn_res_n4 \
  --training.local_batch_size 5 \
  --training.global_batch_size 40 \
  --training.seq_len 260 \
  --parallelism.pipeline_parallel_degree 4 \
  --parallelism.pipeline_parallel_schedule Interleaved1F1B \
  --parallelism.pipeline_parallel_layers_per_stage 2 \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 1 \
  ...

# Crashes ~5 seconds into step 2's backward.
```

To bisect whether it's kimi_linear-specific or general, swap the model
to `--module llama3 --config llama3_debugmodel` with the same PP+V+LBS
settings.

## Status

- 2026-05-03: documented here.
- 2026-05-03: sub-agent dispatched to investigate root cause in
  `torchtitan/torchtitan/distributed/pipeline_parallel.py` and the
  pytorch `pipelining/_backward.py` interaction. Output target:
  `additional_found_issues/torchtitan_pp_lbs_backward_INVESTIGATION.md`.
- TBD: file upstream RFC at https://github.com/pytorch/torchtitan/issues
  with a minimal repro that doesn't depend on AttnRes / kimi_linear.
