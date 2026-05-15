# Issue 1: torchtitan PP + Interleaved1F1B + LOCAL_BS ≥ 3 + V ≥ 2 → "Trying to backward through the graph a second time"

> **STATUS (2026-05-03): RESOLVED in our codebase.** The root cause was NOT
> in upstream pytorch PP, NOT in AttnRes, and NOT in the recv-buffer alias
> originally hypothesized. It was the multimodal-training pattern in
> `phase5_vlm_multimodal_sft/train_mm.py`: the projector ran ONCE per step (in
> `post_dataloading_process`) producing `vision_embeds`, then PP chunked
> that tensor per microbatch — every chunk routed grad back to the SAME
> projector grad_fn. Under V≥2+LBS≥3+Interleaved1F1B, mb_0's stage_backward
> freed the projector's saved tensors and mb_1's stage_backward then hit
> "second time".
>
> Fix: detach the projector output before injecting into the PP input
> dict, and override `MultimodalTrainer.forward_backward_step` to do a
> single deferred `torch.autograd.backward(vision_embeds_orig,
> vision_embeds_leaf.grad)` at the end of each step, replaying the summed
> grad through the projector with a single autograd traversal.
>
> Two diagnostics confirmed the root cause before the fix:
> 1. Patching `_backward.stage_backward` to pass `retain_graph=True` lets
>    the same buggy config train cleanly (3 steps, loss 12.24→12.14→12.09).
> 2. Detaching `vision_embeds` (no projector training, but graph severed)
>    also passes 3 steps cleanly.
>
> The proper fix produces the same loss curve as `retain_graph=True`
> (projector trains correctly) at lower memory (11.4 GiB vs 16.4 GiB).
>
> See `phase5_vlm_multimodal_sft/train_mm.py:MultimodalTrainer.post_dataloading_process` and
> `forward_backward_step` for the implementation.

> **This is a software BUG, not a hardware bottleneck.** The error fires
> inside the autograd graph traversal logic in pytorch's `pipelining/_backward.py`,
> independent of which interconnect (PCIe / NVLink / IB) is used. The
> repro would crash identically on an H100 NVLink box. Distinguish from
> the PCIe-bandwidth observation in `phase7_nccl_traffic_catalog/THROUGHPUT_BOTTLENECK_ANALYSIS.md`,
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
torchrun --nproc_per_node=8 -m phase5_vlm_multimodal_sft.train_mm \
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

## H1 vs H2 disambiguation (2026-05-03 evening)

Empirical experiment to determine whether the bug is upstream-pytorch
(H1) or in our experiment code (H2).

**Smoke 1 — baseline kimi backbone (no AttnRes), same broken recipe**
```
FLAVOR=kimi_linear_436m_baseline      # NO AttnRes, NO cache adapter
FSDP=2 PP=2 TP=2 V=2  LBS=14 GBS=112  # the LBS≥3+V≥2 broken combo
→ ✅ 10 steps clean, loss 7.88 → 7.43, no crash
```

**Smoke 2 — AttnRes flavor, naive PP (cache adapter disabled)**
```
FLAVOR=kimi_linear_436m_block_attn_res_n4  # AttnRes ON
ADAPTER=0                                   # cache adapter OFF (naive PP)
FSDP=2 PP=2 TP=2 V=2  LBS=14 GBS=112
→ ❌ "RuntimeError: Trying to backward through the graph a second time"
```

**Smoke 3 — AttnRes flavor + cache adapter (the original repro)**
```
FLAVOR=kimi_linear_436m_block_attn_res_n4  # AttnRes ON
ADAPTER=1                                   # cache adapter ON
FSDP=2 PP=2 TP=2 V=2  LBS=14 GBS=112
→ ❌ Same crash
```

**Verdict: H2 confirmed, but narrowed further** — the bug is in our
AttnRes path (kimi_linear AttnResModel + block_attn_res aggregator),
**not** in the cache adapter (which is only the optimization layer
on top of naive PP), and **not** in upstream pytorch/torchtitan PP
itself. The baseline kimi backbone with the EXACT broken recipe runs
clean for 10 steps, including step 2 backward where the crash always
fires under AttnRes.

This means:
- Filing an upstream RFC at pytorch/pytorch is **not warranted** —
  the upstream PP path handles V≥2+LBS≥3 correctly when given a
  well-behaved model.
- The fix lives entirely inside our fork, in
  `torchtitan/experiments/{attn_res,kimi_linear}/`. Specifically the
  AttnRes-introduced operations (`block_attn_res`,
  `KimiAttnResDecoderLayer.forward` block-list/partial-block threading,
  `KimiLinearAttnResModel.forward` PP boundary stack/unstack) must be
  creating a tensor alias that V≥2 + LBS≥3 + Interleaved1F1B exposes.
- The cache adapter is a passenger; both adapter ON and OFF crash
  identically.

**Hypotheses for the AttnRes-internal alias (next investigation):**
- The `partial_block` is the recv-buffer or embed_tokens output. It
  enters every layer's `block_attn_res` as input AND gets accumulated
  across layers into the new block list. If a partial_block tensor
  retains an autograd reference to its source (recv buffer / embed
  output) across multiple layer calls in V≥2 schedule, the saved-for-
  backward set might cross-link microbatches.
- The `attn_res_proj.weight` (Linear(d → 1) zero-init) is shared across
  all microbatches as a parameter. Zero-init + multiple microbatches
  feeding through it might create degenerate gradient state that the
  schedule's interleaved backward mishandles.
- The `final_attn_res_proj` + `final_attn_res_norm` at the last PP
  stage operate on the full `block_list + [partial_block]`; under V≥2
  on the last stage device, two microbatches' final aggregations may
  share a leaf reference.

**Status:** sub-investigation deferred. Practical path forward: keep
v10 / production runs at LBS=1 + V=2 (proven safe), file this as
internal AttnRes-fork issue (not upstream), root-cause when GPU time
permits.

## Why hasn't this been reported widely if PP+V≥2+LBS≥2 is "normal"?

User's question, valid skepticism. PP V=2 LBS=2+ should be a routine
production setup; if every torchtitan production user hit this, there'd
be a flood of issues. Several plausible reasons it hasn't been:

1. **torchtitan integration tests run with LBS=1**. The Llama3 / DSv3
   PP smoke tests in `torchtitan/tests/integration_tests/models.py` use
   `local_batch_size=1` (we verified — search for "DeepSeek V3 PP+FSDP+TP"
   block, the runner config does not set LBS>1). The CI never exercises
   the LBS≥2 path under V≥2 Interleaved1F1B with a meaningfully large
   model.

2. **The pytorch tutorial** (`pipelining_tutorial.html`) demonstrates
   PP at LBS=1, never LBS≥2. Most newcomers calibrate to LBS=1 because
   that's what the docs show.

3. **Many production runs use compile path which incidentally fixes
   the alias.** When stages are wrapped with `torch.compile`, dynamo
   emits a `clone()` or `detach()` at FX graph boundaries that breaks
   the recv-buffer aliasing as a side effect. The bug is exposed only
   when stages run uncompiled (eager) and the recv buffer flows
   straight into the autograd graph as a leaf input.

4. **The bug is conditional on the schedule's exact microbatch
   pipeline timing**. With small N_microbatches, there's no room for
   the next-step's irecv to overlap with this step's backward, so the
   buffer overwrite happens after the alias is consumed. The window
   opens at LBS≥2 (multiple microbatches per dp_rank that fight over
   the same recv slot across step boundaries) AND V≥2 (interleaved
   schedule that issues backward after later forwards within the same
   step).

5. **Our specific reproduction is the kimi_linear AttnRes flavor with
   a 12-layer KimiAttnResDecoderLayer model.** We have NOT yet
   confirmed the bug fires on a fresh torchtitan checkout with
   `--module llama3 --config llama3_8b_full` at LBS=2 PP=2 V=2. That
   reproduction is a 5-minute torchrun and would establish the bug as
   independent of our experiment code. **TODO before filing upstream.**

6. **Maybe it WAS fixed in pytorch nightly** but our box is on stable
   2.11.0+cu130. Worth diffing
   `pytorch/main:torch/distributed/pipelining/stage.py` against ours
   before filing — the fix might already be in the trunk and we just
   need a backport / version bump.

In short: not seeing prior reports doesn't mean the bug is fake; it
likely means most users either (a) stayed at LBS=1, (b) used compiled
stages that mask the bug, or (c) on a different pytorch trunk where
the fix already landed. We will reproduce on vanilla llama3 +
upstream-trunk torch before opening the issue.

## Status

- 2026-05-03: documented here.
- 2026-05-03: sub-agent investigation completed (committed at
  `1ad310e`). Root cause: `args_recv_info[chunk_id].buffer` is
  allocated once in `_prepare_forward_infra` and reused across every
  step; `forward_one_chunk` stores the live buffer reference into
  `fwd_cache[chunk_id]` as `input_values`; the next step's irecv
  overwrites the buffer; this step's backward walks an autograd graph
  with freed/overwritten saved tensors → crash.
- 2026-05-03: monkey-patch hotfix (`phase6_upstream_pr_prep/torchtitan_pp_backward_hotfix.py`)
  attempted: clone `input_values` after the original `forward_one_chunk`
  computes the output. **Did NOT work**. Cloning AFTER forward breaks
  PP's gradient-recovery path: gradient is accumulated on the original
  recv buffer (the autograd leaf the output graph saved), but PP's
  `get_bwd_send_ops` reads `.grad` from `input_values` (now a clone
  whose `.grad` stays None). New error at step 1 backward:
  `RuntimeError: [N] for chunk 0 has gradients None and is expecting
  to send gradients to stage M`. Hotfix reverted to a no-op pass-through
  for posterity.
- A real fix needs one of:
  - **(A)** Per-step recv buffer allocation: make `_prepare_forward_infra`
    allocate a fresh buffer per (step, chunk_id) instead of reusing.
    Costs `2 × max_in_flight × buffer_size` extra memory, but the
    autograd alias goes away cleanly. **The only viable fix path** —
    invasive but correct.
  - **(B)** ❌ Move buffer cloning earlier: clone INSIDE
    `_retrieve_recv_activations` so the model's forward operates on
    the clone. **Tested 2026-05-03 via vendored stage.py overlay
    (phase6_upstream_pr_prep/torchtitan_pp_patches/), failed.** The clone is non-leaf
    and gradient at non-leaf doesn't reach `bwd_cache[chunk_id]` via
    `stage_backward()` — `grads_input` ends up `None` for the clone
    entry, then `get_bwd_send_ops` hits the same secondary error
    `"[N] for chunk 0 has gradients None and is expecting to send
    gradients to stage M"` as the post-cache clone variant.
  - **(C)** Modify backward send to read `.grad` from the recv buffer
    (which IS where gradient lands today) instead of from input_values.
    Smallest API surface change but coupling backward send to the recv
    buffer object identity is fragile.
  Each requires a torch core PR. **Two clone-based attempts both
  fail because PP's get_bwd_send_ops fundamentally expects gradient
  to be reachable through the leaves it stored at forward time —
  any buffer remap mid-flight breaks that contract.** The right
  patch is option (A): per-step buffer pool. Beyond what a runtime
  monkey-patch / single-file vendor can express; needs upstream PR.
- Until upstream fix lands: workaround is **LBS=1 for any PP+V≥2 run**.
  v10 / A3 alignment / A2 alignment all stable at LBS=1.
- TBD: minimal repro on vanilla torchtitan llama3 (no AttnRes / kimi_linear)
  to confirm the bug is upstream-only. Then file
  https://github.com/pytorch/pytorch/issues/new with the per-step
  buffer-reuse race trace + the failed-clone-after experiment as
  evidence that fixing it requires more than `.clone()`.
