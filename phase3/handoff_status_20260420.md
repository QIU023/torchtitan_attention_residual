# Phase 3 adapter-debug handoff — 2026-04-20 01:00

Status of the four issues tracked in-chat while the 8-GPU naive PP smoke
was passing and before launching the adapter A/B. Captured here so the
next session can continue without re-parsing chat scrollback.

## Issue 1 — `K_s=0` empty-commit assert — **FIXED**
Root cause: launcher runs `layers_per_stage=1` → 16 virtual stages; odd
virtual stages never cross `is_block_start`, so when
`_return_only_new_blocks=True` sliced off the prefix it produced an
empty list and tripped `model.py:219`'s assertion.

Fix: `model.py:217-232` now returns `partial.new_zeros((0, *partial.shape))`
instead of asserting. P2P shape stays static (zero first dim), and the
adapter's `unstack_blocks` naturally yields `[]`. New test
`test_return_only_new_blocks_empty_commit` locks the contract.

## Issue 2 — `id(partial)` microbatch key doesn't cross P2P — **FIXED**
Root cause: NCCL allocates fresh receive buffers on the consumer side,
so `id(partial_out_s)` on the producer can never equal
`id(partial_recv_{s+1})` on the consumer. The per-microbatch cache
therefore missed on every middle stage, and the wrapped model saw only
the immediately-previous stage's new blocks — never the full accumulated
prefix. Numerics would have diverged from naive PP.

Fix (in `pipeline_adapter.py`): monkey-patch `PipelineStage.forward_one_chunk`
and `backward_one_chunk` on each wrapped stage so the schedule-owned
`fwd_chunk_id` / `bwd_chunk_id` is stashed in an adapter-keyed
thread-local before the submod is invoked. That integer — not a Python
`id()` — is now the cache key. Forward and backward run on the same
thread synchronously, so autograd hooks read the key reliably even
during backward. Key is stable across P2P because it's derived from
scheduler state, not tensor identity.

## Issue 3 — backward grad send-back not wired — **FIXED**
Root cause: `_register_grad_accumulators` accumulated per-block grads
into `_cache._grad_accum`, but nothing popped them and sent them back
to the producer stage. Once Issue 2 was fixed the consumer gained real
cached-prefix blocks, so those grads now need to reach the producer.

Fix (in `pipeline_adapter.py`): two `torch.autograd.Function` classes.

- `_SendBlockGradsBack` (consumer side): forward is a pass-through that
  repackages the cached prefix into the consumer's autograd graph;
  backward issues a batched `dist.isend` of the per-block grads back to
  the producer stage over the pipeline process group.
- `_RecvBlockGradsFromConsumers` (producer side, symmetric): wraps the
  blocks this stage committed; its backward does one `dist.irecv` per
  consuming stage, sums them, and adds them into the producer's local
  grad.

Unit tests for the mb-indexing patch shape are in the new
`tests/test_pipeline_adapter.py` (`TestMbIndexThreading`).

## Issue 4 — launcher/config comment mismatch — **FIXED**
Configs in `experiments/attn_res/__init__.py` and `config_registry.py`
described the `L16_n8` variant as `layers_per_stage=2, VP=2 → 8 stages`,
but the launchers run `layers_per_stage=1` (16 virtual stages, 2
chunks/rank). Agent pass updated docstrings and inline comments with
the explicit virtual-stage arithmetic:
`(n_layers=16 + first_less=0 + last_less=0) / layers_per_stage=1 =
16 virtual stages ÷ PP=8 = 2 chunks/rank` under Interleaved1F1B.

## Test state
- `pytest torchtitan/experiments/attn_res/tests/ -q` → **30 passed**
  (up from 15 before the fixes).

## Files touched this session
Fork (submodule, branch `attention_residual_dev`):
- `torchtitan/experiments/attn_res/model.py` — empty-commit shape fix.
- `torchtitan/experiments/attn_res/pipeline_adapter.py` — mb-index
  threading + grad send-back autograd.Functions + warnings removed.
- `torchtitan/experiments/attn_res/__init__.py` — L16_n8 comment
  aligned with LPS=1.
- `torchtitan/experiments/attn_res/config_registry.py` — docstring
  aligned with LPS=1.
- `torchtitan/experiments/attn_res/tests/test_attn_res.py` — empty-
  commit test added + docstring fix.
- `torchtitan/experiments/attn_res/tests/test_pipeline_adapter.py` —
  NEW: mb-index threading tests.

Workspace:
- `phase3/adapter_design.md` — open-unknowns section updated to reflect
  resolved items and chosen implementation strategy.
- `phase3/handoff_status_20260420.md` — this file.

## Next action
All four known blockers are resolved. Launch the adapter PP smoke and
compare against naive:

```
rm -rf phase3/runs/pp8_adapter
source /venv/main/bin/activate
bash phase3/launch_8gpu_adapter.sh          # 1000 steps, adapter ON
python phase3/compare_pp_vs_single.py \
    --single phase2/runs/attn_res/tb \
    --pp    phase3/runs/pp8_naive/tb \
    --pp_cached phase3/runs/pp8_adapter/tb
```

Expectation: adapter's rank-7 TB loss matches naive PP's rank-7 TB loss
within bf16 tolerance; max-abs-diff from `compare_pp_vs_single.py`
should be small.

## Residual risks worth eyeballing
- Monkey-patched `PipelineStage.forward_one_chunk` /
  `backward_one_chunk` bind on a specific torch 2.11 signature. If
  torch refactors, the adapter falls back with an explicit error via
  `_iter_schedule_stages` (same pattern already in the file).
- Grad send-back uses the pipeline process group; it must be created
  before the first backward microbatch. Normal schedule construction
  order handles this, but FSDP reshard under AC rerun was not exercised.
- Gradient dtype: `dist.isend` / `irecv` tensors are bf16 in our
  configs. Compile + flash-attn haven't been stressed yet.
