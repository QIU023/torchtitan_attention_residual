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

1. ~~**Microbatch keying.**~~ **Resolved.** We went with option (c) executed
   inside the experiment: `pipeline_llm_with_cache_adapter` monkey-patches each
   stage's `forward_one_chunk(fwd_chunk_id, ...)` and
   `backward_one_chunk(bwd_chunk_id, ...)` to stash the integer index on a
   thread-local keyed per adapter instance. The adapter's `forward` reads the
   thread-local at entry; autograd backward runs on the same thread shortly
   after, so the same thread-local is still valid when hooks fire. The key is
   a plain int, so it survives P2P crossings exactly — producer and consumer
   both key on the same `fwd_chunk_id` issued by the schedule. `id(tensor)` is
   no longer used. See
   [`torchtitan/experiments/attn_res/pipeline_adapter.py:_install_mb_index_patch`](../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py).

2. **VP chunk order.** Still live: we read `stage.group_rank` per
   `stage.stage_index` and build a `stage_to_rank` map at setup time so the
   adapter's grad send-back targets the correct rank even under interleaved /
   looped schedules. Needs a real 8-GPU run to confirm.

3. ~~**Backward hook reliability.**~~ **Resolved by sidestepping hooks.** The
   new design uses two `torch.autograd.Function` subclasses instead of
   `register_hook`: `_SendBlockGradsBack` wraps the cached prefix on the
   consumer side (backward = accumulate into `_grad_accum` + `dist.isend` to
   producer), `_RecvBlockGradsFromConsumers` wraps the emitted blocks on the
   producer side (backward = `dist.irecv` from each consumer stage + sum into
   local grad). AC replay re-runs the whole adapter forward, which re-creates
   the Function's autograd graph — no stale hooks to worry about.

4. **AC interaction.** Still live but less fragile: the adapter's forward is
   already idempotent in the sense that a re-forward for the same mb_index
   would re-populate `_cache` with the same tensors (because the schedule
   gives the same `fwd_chunk_id`). The `_SendBlockGradsBack` autograd graph
   gets rebuilt cleanly on re-forward. Needs a concrete AC-enabled 8-GPU run
   to confirm.

5. **FSDP reshard + adapter.** Still live, same as before. The adapter wraps
   the model so FSDP's pre-fwd hooks fire on the adapter's `__call__`. No
   design change here.

### Grad send-back protocol (solves Bug #3 in pipeline_adapter)

Tag scheme: each cached block has a producer stage `p` and an index-within-
producer `b`. Its P2P tag is `_grad_tag_base(mb, p) + b`
(`_grad_tag_base` reserves 1024 tags per `(mb, producer)` pair). Consumer
(stage `c`) posts `dist.isend(grad, dst=rank(p), tag=tag)`. Producer (stage
`p`) pre-posts `dist.irecv(buf, src=rank(c), tag=tag)` once per consumer
stage, sums all N buffers into a single `extra` tensor, and adds it to the
local grad in its own backward via
`_RecvBlockGradsFromConsumers.backward`. This makes producer-side grad
equal to `(local_autograd_grad_from_partial_next) + sum_over_consumers(grad_i)`,
which is exactly the naive PP value.

CPU unit tests run with `group=None`; both Functions silently skip the
P2P ops but still exercise the accumulation logic (`_SendBlockGradsBack`
writes into `_cache._grad_accum` unconditionally, so tests can assert on
the payload without NCCL).

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
