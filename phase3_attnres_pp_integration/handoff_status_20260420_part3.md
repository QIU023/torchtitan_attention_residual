# Phase 3 adapter — session 3 log (2026-04-20 08:10 → 09:00)

Continuation of `handoff_status_20260420_part2.md`. Covers the
`_LocalCacheAugment` / `_LocalCacheCapture` implementation and the
unresolved double-backward it hits on 8-GPU despite passing CPU tests.

## TL;DR

- Replaced the `retain_graph=True` hack with two local-only
  `torch.autograd.Function` classes, per the design proposed at
  end of part 2.
- CPU tests pass (41 / 41 green, including a fresh 4-stage P=2 V=2
  adapter canary `test_backward_grad_equivalence_4stage_vp2`).
- 8-GPU smoke run still hits **`RuntimeError: Trying to backward
  through the graph a second time`** at rank 0 stage 0's
  `stage_backward` — the exact same failure the retain_graph hack was
  papering over.
- Best current hypothesis: both Functions return `block_tensor`
  directly (same Python object as input) which makes autograd's
  grad_fn bookkeeping ambiguous under real PP scheduling. A partial
  fix (return `block_tensor.view(block_tensor.shape)` to force a
  distinct Tensor wrapper) was edited into the file but the session
  ended before it was retested.
- Fork HEAD: `attention_residual_dev@690b33a` (`_LocalCache*` lands).
  Workspace HEAD: `main@44e6b96`. Local working tree has uncommitted
  edits adding the `.view(...)` fix to both Function `forward`s
  — NOT committed, NOT pushed.

## What shipped (commit `690b33a`)

Two new `torch.autograd.Function` classes in `pipeline_adapter.py`:

- `_LocalCacheAugment(block_tensor, slot_key, rank_cache)`:
  - `forward` returns the tensor, stashes `slot_key` + `rank_cache`
    on ctx.
  - `backward(grad)` pops any captured grad at `slot_key`, returns
    `grad + captured` for the tensor input, `None` for the other two.
  - Applied at producer emission in `_finish_forward`: each newly
    committed block is wrapped BEFORE being appended to the rank
    cache and BEFORE being used in the outgoing delta.

- `_LocalCacheCapture(block_tensor, slot_key, rank_cache)`:
  - `forward` identity pass-through + ctx stash.
  - `backward(grad)` calls `rank_cache.capture_grad(slot_key, grad)`
    (accumulating sum) and returns `None` for the tensor input,
    intentionally stopping autograd from propagating upstream.
  - Applied at consumer read in `_forward_delta`, but ONLY when the
    cached block's metadata says `producer_rank == self.pp_rank`
    (own-rank commit). Recv-originated blocks stay unwrapped — their
    grad rides the `recv_delta_tensor` back through PP's built-in
    `SEND_B`.

`RankLocalCache` grew a `_captured_grads` dict plus helpers
(`capture_grad`, `pop_grad`, `has_captured_for_mb`). `drop` sweeps
stale slots keyed on the dropped mb.

`_install_mb_index_patch.patched_bwd` has the retain_graph override
completely removed; it's back to a plain `try/finally` that stashes
and clears the mb index.

Dead code and dead assertions (especially every leftover reference
to `add_grad` / `pop_grads` / the removed cache slots) were deleted.
`pipeline_adapter.py` stays under 900 lines.

### Tests (41 / 41 green on CPU)

Five new tests in `tests/test_pipeline_adapter.py`:

- `test_local_cache_capture_blocks_backward_propagation`
- `test_local_cache_augment_adds_captured_to_incoming_grad`
- `test_multi_consumer_augment_sums_across_captures` (V>2 case)
- `test_producer_param_grad_equivalence_to_naive`
- `test_backward_grad_equivalence_4stage_vp2` (P=2 V=2 adapter canary
  exercising the full same-rank own-commit cache-read path)

Plus the existing canaries `test_forward_delta_numerics_2stage` and
`test_backward_grad_equivalence_2stage` remain green.

## What broke on 8-GPU

Launcher: `phase3_attnres_pp_integration/launch_8gpu_adapter.sh` targeting the new
`llama3_175m_attn_res_L16_n8` config (renamed from `150m` in commit
`328352f` this session; 174M real params, previously misnamed).

Crash trace (rank 0, stage 0 under ScheduleInterleaved1F1B):

```
File "torch/distributed/pipelining/_backward.py", line 370, in stage_backward
    torch.autograd.backward(...)
File "torch/autograd/__init__.py", line 381, in backward
    _engine_run_backward(...)
File "torch/autograd/graph.py", line 869, in _engine_run_backward
    return Variable._execution_engine.run_backward(...)
RuntimeError: Trying to backward through the graph a second time
    (or directly access saved tensors after they have already been freed).
```

The fork's P=2 V=2 CPU canary passes this exact pattern algebraically
(producer's `param.grad` matches naive to 1e-5). So the bug only
surfaces under the real schedule + gloo/NCCL backend + torchtitan's
`stage.backward_one_chunk` code path.

## Best current hypothesis + partial fix

Both Function `forward`s did:
```python
@staticmethod
def forward(ctx, block_tensor, slot_key, rank_cache):
    ctx.slot_key = slot_key
    ctx.rank_cache = rank_cache
    return block_tensor
```

Returning the same Python tensor object from `autograd.Function.forward`
is undefined by PyTorch: the output "should be" a node whose grad_fn
is the Function's backward, but the input tensor already had its own
grad_fn. Under simple CPU autograd pytorch silently picks the right
one; under the full PP schedule (which caches forward outputs and
re-uses them across `stage.backward_one_chunk` calls) the ambiguity
is suspected to manifest as an incorrect second traversal of the
producer stage's forward graph.

The partial fix in the working tree (uncommitted) replaces both
`return block_tensor` lines with:
```python
return block_tensor.view(block_tensor.shape)
```
which returns a distinct Tensor wrapper sharing storage — forcing
a fresh autograd node with the Function's backward as grad_fn and no
conflict with the input's grad_fn.

Tests were NOT re-run after this edit because the session ended. Next
session must:
1. Run `pytest torchtitan/experiments/attn_res/tests/ -q` — expect
   41 / 41 green still.
2. Relaunch 8-GPU and see whether the double-backward disappears.

## Debug on cheaper hardware

Yes, this is reproducible on **2 GPUs** (or even 1 GPU with torch's
fake process-group mode if we ever wire it up). The design knob that
triggers the same-rank own-commit cache-read path is `V >= 2`, and the
smallest Interleaved1F1B-legal layout is `PP=2 V=2` → 4 virtual stages
on 2 ranks. Every autograd-ordering subtlety we chased under `PP=8
V=2` is equally exposed at `PP=2 V=2`.

### Minimal 2-GPU debug recipe (proposed)

Add a tiny config to `torchtitan/experiments/attn_res/`:
- `attn_res.py`: nothing (no new model code).
- `__init__.py`: register a flavor `tiny_attn_res_L4_n2` with
  `n_layers=4`, `num_blocks=2`, `dim=128` (or similar), `enable_weight_tying=False`.
- `config_registry.py`: `tiny_attn_res_pp2_vp2` Trainer config
  pointing at the new flavor with `steps=50`, `local_batch_size=2`,
  `seq_len=256`.

Launcher (new file `phase3_attnres_pp_integration/launch_2gpu_adapter.sh`):
```
--nproc_per_node 2
--parallelism.pipeline_parallel_degree 2
--parallelism.pipeline_parallel_schedule Interleaved1F1B
--parallelism.pipeline_parallel_layers_per_stage 1
--parallelism.pipeline_parallel_first_stage_less_layers 0
--parallelism.pipeline_parallel_last_stage_less_layers 0
--training.steps 50
--training.local_batch_size 2
--training.global_batch_size 2
--local-ranks-filter 1 --role rank --tee 3
```

With `PP=2 V=2 n_layers=4 num_blocks=2`, rank 0 owns virtual stages 0
and 2 (which both commit blocks because every virtual stage is a
block-start under 1 layer / block). Rank 1 owns 1 and 3. Stage 2 on
rank 0 reads the cached own-commit from stage 0 — the exact same
`_LocalCacheAugment` / `_LocalCacheCapture` interaction as the
`PP=8 V=2` case but with 4 stages instead of 16. If the
`.view(block_tensor.shape)` fix works, this config will exercise it.
If it still crashes, the extra debuggability (only 2 ranks' logs, only
4 stages in the Interleaved schedule trace) makes narrowing cheap.

Why NOT reproduce on 1 GPU with fake_pg: torch's `fake_pg` backend
doesn't fully drive `ScheduleInterleaved1F1B`'s P2P send/recv ops
today — it asserts during `_shape_inference`. Sticking with 2 real
GPUs is the simpler path.

Minimum vast.ai spend for 2x 4090 or 2x 3090 is ~$0.20–0.40/hr,
vs 8x 5090 at ~$6+/hr. For iteration on autograd-ordering bugs, the
cheaper box is the right tool.

## Files touched this session (fork)

- `torchtitan/experiments/attn_res/pipeline_adapter.py` — added
  `_LocalCacheAugment`, `_LocalCacheCapture`, wired them in
  `_forward_delta` / `_finish_forward`, removed retain_graph patch.
  Working tree has uncommitted `.view(...)` edits on both Function
  forwards.
- `torchtitan/experiments/attn_res/tests/test_pipeline_adapter.py` —
  5 new tests, 36 → 41 green.

Historic sibling commit on origin (pulled + resolved in-flight):
- `experiments/attn_res: rename 150M -> 175M to reflect real param
  count` (`328352f`), updates config flavor names and the experiment
  README. Only string renames; the shape of every identifier this
  session worked with already migrated to `175M*` in our working tree.

## Open action items for next session

1. **(highest priority)** Commit the `.view(block_tensor.shape)` edit
   (or a better variant if the idiom is wrong). Run the full fork
   pytest suite. Re-launch 8-GPU and re-read train log — this may
   just work.
2. If (1) still crashes: reproduce on **2 GPUs** with the tiny config
   + launcher above. Add per-stage dbg prints inside both Function
   backwards to see which stage's traversal is the "second" one.
3. If (2) confirms the bug is specifically in `stage.backward_one_chunk`
   + `autograd.Function.forward` returning shared tensors: consider
   rewriting to use a `torch.utils.hooks.BackwardHook` pattern
   instead of a full `autograd.Function` — that side-steps the
   "output is the input" ambiguity entirely.
4. Memory claim verification is still pending — whenever the 1000-step
   run clears, capture rank 7 peak memory and compare to naive's
   6.88 GiB baseline. Expected: ~7 GiB, no longer 11.9 GiB.
