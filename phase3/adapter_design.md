# CrossStageCacheAdapter — design

## What the adapter is

A `nn.Module` that wraps the per-stage AttnRes decoder submodule. Installed
via the experiment's custom `pipelining_fn` —
[`torchtitan/experiments/attn_res/pipeline_adapter.py:pipeline_llm_with_cache_adapter`](../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py) —
which is registered on `ModelSpec.pipelining_fn`. It calls core
`pipeline_llm`, then iterates `schedule._stages` and wraps each
`stage.submod` with `CrossStageCacheAdapter` when
`TORCHTITAN_ATTNRES_CACHE=1`.

**Zero core modifications** — this matches the
`transformers_modeling_backend` experiment's pattern of customizing PP via
the ModelSpec hook.

The adapter makes inter-stage forward bandwidth **constant in stage id** by
caching the blocks each stage has already committed, and sending only the
block(s) newly committed at each boundary.

## Invariants the wrapped model must satisfy

The wrapped model is `AttnResLlama3Model` (experiments/attn_res/model.py):

- Non-last stage forward: takes `(partial_block, blocks_tensor)` and
  returns `(partial_block, stacked_new_blocks_for_this_stage_only)`.
  **This is the key change from Phase 2's model**, which currently returns
  the entire stacked blocks including earlier-stage blocks. See
  "Required model change" below.
- Last stage forward: takes `(partial_block, blocks_tensor)` and returns
  the final logits.
- Each stage knows `num_blocks_committed_before_this_stage` so it can
  index into its cache correctly.

## Forward invariant

| symbol | meaning |
| --- | --- |
| `K_i` | number of blocks committed by stage `i` |
| `M` | number of in-flight microbatches (depends on schedule) |
| `cache[mb_id]` | list of `[B, T, D]` tensors, length `sum(K_j for j<stage_id)` |

On recv at stage `s`:
- Receive `partial_block_s`, `new_blocks_from_stage_s_minus_1`.
- `cache[mb_id] = cache[mb_id] + list(new_blocks_from_stage_s_minus_1)`.
- Hand the wrapped model `blocks = stack(cache[mb_id])` and `partial = partial_block_s`.
- Wrapped model runs layers of stage `s`, returns `(partial_block_s',
  new_blocks_from_stage_s)` of size `K_s`.
- Send `(partial_block_s', new_blocks_from_stage_s)` forward. **Constant size**
  in stage id.

## Backward invariant

Autograd on stage `s` sees:
- Grad flowing into `partial_block_s'` (sent forward to stage `s+1`).
- Grad flowing into each element of `new_blocks_from_stage_s` — from
  every `stage_s+1 ... stage_last-1` that consumed them via the attention
  over blocks. Each later stage's attention weights produce a gradient
  term on the blocks it received.

The adapter responsibility:
1. Before handing the wrapped model the full `blocks` list, it must
   register backward hooks on the cached (from-earlier-stages) portion of
   the blocks so that the grads accumulate into per-block accumulators.
2. When stage `s`'s backward runs across all `M` microbatches, it needs to
   send back: `partial_block_s.grad` (one tensor per microbatch) AND
   `cache[mb_id].grad` — the accumulated gradient for each block that
   arrived at this stage during forward.
3. Send `new_blocks_from_stage_s_minus_1.grad` back to stage `s-1`.

## Model change (implemented)

`AttnResLlama3Model.forward` now supports `_return_only_new_blocks`. When
False (default, Phase 2 behavior), the non-last-stage branch returns the
entire stacked block list. When True (adapter sets it on its wrapped
submod at construction), the branch slices off everything before
`initial_num_blocks` and returns only the blocks committed by the current
stage — which is what lets the adapter keep per-stage send size constant.

The adapter flips the flag in its `__init__` with a `hasattr` probe so
the wrapping is safe against older models that don't expose the flag
(those fall back to naive full-stack behavior with a warning).

Guard in model.py: if `_return_only_new_blocks` is True but the stage
committed zero blocks, an assertion fires. This can happen only if
`num_blocks < num_stages`, which is a config error — fix by increasing
`num_blocks` or decreasing PP stages.

## Open unknowns

Resolve these when the adapter actually boots on 8 GPUs:

1. **Microbatch keying.** `torch.distributed.pipelining` doesn't explicitly
   expose a microbatch id to the submodule. Options: (a) use `id(partial_block)`,
   (b) increment an internal counter and rely on schedule order, (c) patch
   `PipelineStage.forward` to pass an index kwarg. (b) is simplest but
   fragile under any schedule that re-enters forward for re-materialization.
   Start with (a) — activation identity is stable across the forward/backward
   of one microbatch.

2. **VP chunk order.** `PP=8, VP=2` means each rank owns two virtual stages,
   non-contiguous in depth: rank 0 owns stage 0 + 8, rank 1 owns 1 + 9, etc.
   (interleaved). The adapter must cache PER virtual stage, not per rank. Use
   `stage_id` that includes VP, not just `pp_rank`.

3. **Backward hook reliability.** When the schedule's activation recomputation
   (AC or FSDP reshard) reruns forward, does `register_hook` on the cached
   block tensors still fire correctly during backward? We have to test. If it
   doesn't, fall back to explicitly returning `(partial_block, cache_refs)`
   from the adapter's forward and writing an explicit `torch.autograd.Function`
   that owns the grad accumulation.

4. **AC interaction.** `activation_checkpoint.mode=selective` reruns forward
   on a subset. For AttnRes, the rerun must NOT re-receive blocks over P2P
   (they've already arrived); instead, it should hit the adapter's cache.
   This probably means the adapter's `forward` needs to be idempotent for the
   same microbatch id.

5. **FSDP reshard + adapter.** FSDP2 unshards on `__call__`. The adapter wraps
   the model so FSDP's hook fires on the adapter's `__call__` which then
   delegates to the inner model — same number of all_gathers as before. But
   when AC triggers a re-forward, and the adapter shortcuts to the cache, does
   FSDP still properly resharden? Needs explicit test.

## Rollout order (decisions I'm making for the draft)

1. Adapter implements the caching + send-constant-size, and the model
   already supports `_return_only_new_blocks`. Both are off by default —
   the custom `pipelining_fn` is a passthrough when
   `TORCHTITAN_ATTNRES_CACHE` is unset.
2. **First boot**: `TORCHTITAN_ATTNRES_CACHE` unset → naive path, proves
   PP numerics match single-GPU (`compare_pp_vs_single.py`).
3. **Turn on adapter**: `TORCHTITAN_ATTNRES_CACHE=1` → expect identical
   loss to naive PP within bf16 tolerance. Comparing both in the same
   500-step run is the correctness acceptance test.
4. Only after 3 passes, compare comm time: naive should show growing
   send volumes (per-stage bytes grow with stage id), adapter should
   show constant.

If step 3 fails (loss diverges), open unknowns 3 or 4 above are the likely
culprits; document what happened and pivot back to naive PP with growing
tensors (still a valid, if less optimal, PR #2 story).
