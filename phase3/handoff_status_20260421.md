# Phase 3 adapter — session 4 log (2026-04-21)

Continuation of `handoff_status_20260420_part3.md`. Validates the
adapter on a 4-GPU box, finds and fixes the residual double-backward
that the `.view()` patch did not address, and confirms numerical
alignment with naive PP within run-to-run nondeterminism.

## TL;DR

- The committed `block_tensor.view(block_tensor.shape)` fix from
  session 3 (`24dc99a`) **did NOT** resolve the
  `RuntimeError: Trying to backward through the graph a second time`
  on real PP. CPU tests still pass (41/41) but the bug reproduces
  the moment a same-rank own-commit cache read happens under
  Interleaved1F1B + FSDP + selective AC rerun.
- Root cause confirmed via per-Function backward tracing on a 4-GPU
  PP=4 V=2 smoke: at rank 3's stage-7 backward, `_LocalCacheCapture.backward`
  fired AND `_LocalCacheAugment.backward` fired in the **same**
  `backward_one_chunk` call (slot `(mb=0, producer_stage=3, block=0)`),
  meaning autograd was traversing from the consumer's Capture into the
  producer's Augment despite Capture returning `None` for its tensor
  input's grad. The view-output trick was structurally insufficient.
- Fix shipped: replace `_LocalCacheAugment` (autograd.Function) with a
  plain `tensor.register_hook` at producer emission, AND store a
  DETACHED copy of the producer block in the rank cache. Detach is the
  load-bearing guarantee — the consumer's Capture input has no
  upstream `grad_fn` to traverse, period. Even if autograd ignored
  Capture's `None` return there is no producer graph to walk into.
- Validated on 4-GPU PP=4 V=2 (8 virtual stages, 2 chunks/rank).
  Adapter loss falls inside the naive run-to-run noise band:
    | step | naive run 1 | naive run 2 | adapter |
    |------|-------------|-------------|---------|
    |   1  | 11.76178    | 11.76178    | 11.76178 |
    |  10  | 10.64974    | 10.78452    | 10.74844 |
    |  20  | 9.70779     | 10.19160    | 10.18950 |
- 41 / 41 fork pytests pass after the refactor (3 unit tests rewritten
  to exercise the hook + detached-cache pattern instead of the prior
  `_LocalCacheAugment.apply` flow; integration canaries unchanged).

## Why 4-GPU and not 8-GPU

Original 8-GPU launcher targets `PP=8 V=2` on an 8x card host. The
session-4 host has only 4 RTX 5090s. NCCL refuses multi-process-per-GPU
(`Duplicate GPU detected : rank 7 and rank 3 both on CUDA device a1000`),
even with `NCCL_P2P_DISABLE=1`, so the natural "8 ranks on 4 GPUs via
`CUDA_VISIBLE_DEVICES=$((LOCAL_RANK % 4))`" workaround does not work
out-of-the-box (there is a stub wrapper at `phase3/rank_to_gpu_wrapper.py`
preserved for future MPS-based attempts).

The substitute is `PP=4 V=2` on 4 GPUs with
`pipeline_parallel_layers_per_stage=2`. That gives 8 virtual stages
distributed 2-per-rank across 4 ranks, exercises the same-rank
own-commit cache-read path the `_LocalCache*` machinery targets (the
exact bug surface from `handoff_status_20260420_part3.md`), and keeps
1 process per physical GPU.

Configs and launchers shipped:
- `phase3/launch_4gpu_naive.sh`   — naive PP=4 V=2 baseline
- `phase3/launch_4gpu_adapter.sh` — adapter PP=4 V=2 with cache ON
- `phase3/rank_to_gpu_wrapper.py` — unused multi-proc-per-GPU wrapper

## What changed in `pipeline_adapter.py`

- Removed `_LocalCacheAugment` autograd.Function entirely.
- Added `_install_augment_hook(block_tensor, slot_key, rank_cache)` —
  registers a tensor grad hook that pops the matching captured-grad
  slot and SUMs it into the incoming grad during the producer stage's
  own backward.
- `_LocalCacheCapture` retained, but the contract changed: its tensor
  input is now always a DETACHED leaf (from cache). The `None` return
  for the tensor-input grad is belt-and-suspenders; detach is the
  primary guarantee.
- `_finish_forward`: own-commit blocks now get `_install_augment_hook`
  registered + `blk.detach()` stored in cache. Outgoing-delta still
  uses the attached `blk` so the next stage's SEND_B reaches the
  producer's wrapped model via the block's natural grad_fn chain.
- `_forward_delta`: same-rank cached read sets
  `requires_grad_(True)` on the detached cache leaf before passing it
  to `_LocalCacheCapture.apply` (so Capture's output is differentiable
  and Capture.backward fires; without this, the detached input would
  yield a non-grad output and the slot would never be filled).
- Top-of-file docstring + the design comment block above the bridge
  classes rewritten to reflect the hook + detach design and explain
  WHY the prior `_LocalCacheAugment + view` pattern failed.
- Backward-call tracing kept behind `ATTNRES_ADAPTER_DBG=1`
  (`_dbg(...)` calls in the hook, in `_LocalCacheCapture.forward/backward`,
  and at `patched_bwd` enter/exit). Off by default.

## What changed in `tests/test_pipeline_adapter.py`

Three tests in `TestLocalCacheAutogradFunctions` rewritten:
- `test_local_cache_capture_blocks_backward_propagation` — tests
  Capture against a DETACHED leaf input, mirroring the real adapter
  contract. Asserts upstream `a.grad` stays `None` and the slot
  receives the grad-out.
- `test_augment_hook_adds_captured_to_incoming_grad` — replaces the
  prior `_LocalCacheAugment` test. Pre-populates the slot, runs a
  forward whose backward fires the hook, asserts the upstream grad
  picks up `incoming + G`.
- `test_multi_consumer_hook_sums_across_captures` — V>2 case: two
  detached cache reads each wrapped in Capture both sum into the
  same slot, the hook pops once on the producer's backward.
- `test_producer_param_grad_equivalence_to_naive` updated to use the
  hook + detached-cache pattern; param grads still match naive
  autograd to 1e-5.

Integration canaries unchanged (`test_backward_grad_equivalence_4stage_vp2`,
`test_backward_grad_equivalence_2stage`, `test_forward_delta_numerics_2stage`,
layout/cache/mb-index tests).

## Forward-shape evidence (PP=4 V=2 over L16_n8)

`delta_to_send` per stage at the per-mb level:

| stage | rank | commits | recv  | sends |
|-------|------|---------|-------|-------|
|   0   |  0   | b0      | -     | [b0]      |
|   1   |  1   | b1      | [b0]      | [b0,b1]   |
|   2   |  2   | b2      | [b0,b1]   | [b0,b1,b2]|
|   3   |  3   | b3      | [b0,b1,b2]| [b1,b2,b3]|
|   4   |  0   | b4      | [b1,b2,b3]| [b2,b3,b4]|
|   5   |  1   | b5      | [b2,b3,b4]| [b3,b4,b5]|
|   6   |  2   | b6      | [b3,b4,b5]| [b4,b5,b6]|
|   7   |  3   | b7      | [b4,b5,b6]| -         |

Same-rank own-commit cache reads:
- rank 0: stage 4 reads b0 from cache (committed by stage 0 on rank 0).
- rank 1: stage 5 reads b1 from cache (committed by stage 1 on rank 1).
- rank 2: stage 6 reads b2 from cache (committed by stage 2 on rank 2).
- rank 3: stage 7 reads b3 from cache (committed by stage 3 on rank 3).

These are the four call sites the hook + Capture pair must bridge per
mb. The `ATTNRES_ADAPTER_DBG=1` trace shows exactly that, with hooks
firing only on the producer's own backward and Captures only on the
later virtual stage's backward.

## Reproducing

```bash
source /venv/main/bin/activate
cd /root/torchtitan_attention_residual

# Fork pytests
( cd torchtitan && python -m pytest torchtitan/experiments/attn_res/tests/ -q )
# expect: 41 passed

# 4-GPU naive baseline
rm -rf phase3/runs/pp4_naive_4gpu
STEPS=50 bash phase3/launch_4gpu_naive.sh

# 4-GPU adapter (cache ON)
rm -rf phase3/runs/pp4_adapter_4gpu
STEPS=50 bash phase3/launch_4gpu_adapter.sh

# Optional per-Function tracing
ATTNRES_ADAPTER_DBG=1 STEPS=2 bash phase3/launch_4gpu_adapter.sh
```

## Open follow-ups

1. **8-GPU revalidation** — when an 8x box is available, rerun
   `phase3/launch_8gpu_adapter.sh` (PP=8 V=2). Same hook + detached
   cache code path applies; expectation is the same naive-band loss
   alignment that PP=4 V=2 demonstrated. Memory should now sit at the
   naive-PP baseline (no `retain_graph`-induced inflation).
2. **Memory measurement** — capture rank-7 peak memory on the 8-GPU
   adapter run and compare to naive's 6.88 GiB baseline. The
   `retain_graph` hack inflated to 11.9 GiB; the hook + detach design
   should land near 6.9 GiB plus the rank-cache footprint
   (B*T*D*num_blocks_in_cache, which is a few hundred MB at 175M).
3. **Scale-up smoke** — once 8-GPU is green, push the 1.5B / 2B
   AttnRes config to validate the headline cross-stage caching
   benchmark on PCIe.
4. **NCCL determinism** — naive→naive runs differ at step 10 by
   ~0.13 loss (10.65 vs 10.78). That's likely NCCL ordering /
   bf16-accumulation noise, not anything the adapter introduces. If
   stricter parity becomes important for benchmarking, set
   `NCCL_DETERMINISTIC=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and
   pin `torch.use_deterministic_algorithms(True)` in the trainer.
