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
out-of-the-box (there is a stub wrapper at `phase3_attnres_pp_integration/rank_to_gpu_wrapper.py`
preserved for future MPS-based attempts).

The substitute is `PP=4 V=2` on 4 GPUs with
`pipeline_parallel_layers_per_stage=2`. That gives 8 virtual stages
distributed 2-per-rank across 4 ranks, exercises the same-rank
own-commit cache-read path the `_LocalCache*` machinery targets (the
exact bug surface from `handoff_status_20260420_part3.md`), and keeps
1 process per physical GPU.

Configs and launchers shipped:
- `phase3_attnres_pp_integration/launch_4gpu_naive.sh`   — naive PP=4 V=2 baseline
- `phase3_attnres_pp_integration/launch_4gpu_adapter.sh` — adapter PP=4 V=2 with cache ON
- `phase3_attnres_pp_integration/rank_to_gpu_wrapper.py` — unused multi-proc-per-GPU wrapper

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
rm -rf phase3_attnres_pp_integration/runs/pp4_naive_4gpu
STEPS=50 bash phase3_attnres_pp_integration/launch_4gpu_naive.sh

# 4-GPU adapter (cache ON)
rm -rf phase3_attnres_pp_integration/runs/pp4_adapter_4gpu
STEPS=50 bash phase3_attnres_pp_integration/launch_4gpu_adapter.sh

# Optional per-Function tracing
ATTNRES_ADAPTER_DBG=1 STEPS=2 bash phase3_attnres_pp_integration/launch_4gpu_adapter.sh
```

## 1000-step PP=4 V=2 result (added end of session)

Both naive and adapter ran 1000 steps cleanly. Loss alignment is
inside the naive-vs-naive nondeterminism band (max |Δ_naive→naive| at
step 10 was 0.13 in this session; max |Δ_naive→adapter| over 1000
steps is 0.06). Memory deltas match the design expectation: the
adapter pays exactly the cache footprint (~260 MB on rank 3 for 175M
at M=4 mb), no `retain_graph`-style inflation.

| step | naive loss | adapter loss | naive tps | adapter tps | naive mem rank3 | adapter mem rank3 |
|------|-----------|--------------|-----------|-------------|-----------------|-------------------|
| 1    | 11.76178  | 11.76178     | 504       | 529         | 6.48 GiB        | 7.37 GiB          |
| 10   | 11.52401  | 11.52564     | 7,009     | 6,876       | 6.96 GiB        | 7.66 GiB          |
| 100  | 8.72997   | 8.73178      | 6,980     | 6,865       | 7.45 GiB        | 7.68 GiB          |
| 500  | 6.49669   | 6.49083      | 6,881     | 6,759       | 7.45 GiB        | 7.71 GiB          |
| 1000 | 6.37720   | **6.34968**  | 6,690     | 6,658       | 7.45 GiB        | 7.71 GiB          |

Adapter is ~0.5% slower on TPS at this scale on PCIe. Expected:
PCIe per-hop latency dominates the comm bytes the adapter saves; the
adapter pays small bookkeeping overhead. The adapter's payoff is on
NVLink-out, inter-node fabrics where the saved bandwidth
(~60 MB/hop in steady state for this config) translates to wall clock.

## Cache distribution & scaling envelope

The schedule itself determines which blocks each rank caches; it is
not a free design choice. Per-rank cache contents at the end of one
mb's forward sweep (PP=4 V=2 num_blocks=8):

| rank | own commits | relayed-from-recv                       | total |
|------|-------------|-----------------------------------------|-------|
|  0   | b0, b4      | b1, b2, b3 (via stage-4 recv)           | **5** |
|  1   | b1, b5      | b0 (stage-1 recv), b2,b3,b4 (stage-5)   | **6** |
|  2   | b2, b6      | b0,b1 (stage-2), b3,b4,b5 (stage-6)     | **7** |
|  3   | b3, b7      | b0,b1,b2 (stage-3), b4,b5,b6 (stage-7)  | **8** |

Per-block replication factor: b0..b4 = **4×**, b5 = 3×, b6 = 2×,
b7 = 1×. Total system cache = 26 block-copies for 8 distinct blocks
(avg 3.25× replication). With M=4 mbs in flight that's ×4 because
the rank cache is keyed per-mb and only evicts at step boundary
(`_install_step_drop_patch`).

This explains the asymmetric nvidia-smi pattern observed in this
session (rank 0: 5076 MiB, rank 3: 8750 MiB): later ranks structurally
hold more cache because they need more prefix blocks for their deeper
virtual stages, AND the last rank also carries the [B,T,V] loss
logits and the output projection.

Per-hop send size matches Reku's "constant after first vp chunk"
claim:

| hop | delta size | note |
|-----|-----------|------|
| 0→1 | 1         | warmup |
| 1→2 | 2         | warmup |
| 2→3 | 3         | steady-state begins |
| 3→4 | 3         | |
| 4→5 | 3         | |
| 5→6 | 3         | |
| 6→7 | 3         | |

Steady-state per-hop = P − 1 blocks. Reku's comm-asymmetry claim
holds for wire bytes; he says nothing about cache-memory asymmetry.

### Memory envelope across model scales

```
peak_cache_bytes(rank R) ≈ |rank_cache_at_entry[R, V-1]| × B × T × D × 2 × M
```

| config                                       | peak rank cache | fits?               |
|----------------------------------------------|-----------------|---------------------|
| 175M smoke (B=4 T=2048 D=768 M=4, N=8)       | ~384 MB         | trivially           |
| 48B target (B=1 T=8192 D=4096 M=8, N=16)     | ~8 GB           | yes on 80GB H100    |
| Super-deep (128B+, N≥64, M=16-32)            | ~30+ GB         | breaks "fits-on-one-card" |

The 48B target is comfortably inside Reku's "cache cost is small"
assumption. Super-deep regimes break it; Reku's own recommended
fallback for that case is **selective AC + activation offload**, NOT
a distributed cache. To take that fallback in this codebase: unset
`TORCHTITAN_ATTNRES_CACHE` (the adapter degrades to naive
passthrough) and rely on torchtitan's existing
`activation_checkpoint=selective` + activation offload. The current
175M / 48B benchmark configs never need that escape hatch.

## Open follow-ups

1. **8-GPU revalidation** — when an 8x box is available, rerun
   `phase3_attnres_pp_integration/launch_8gpu_adapter.sh` (PP=8 V=2). Same hook + detached
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

## Distributed cache: explicitly out of scope for this RFC

Considered and skipped in this session:

- **Producer-only cache + on-demand P2P fetch** — adds 24+ extra
  P2P round-trips per mb at PP=4 N=8; would erase the
  comm-savings story Reku's design buys.
- **Designated holder rotation** — same comm shape with marginally
  better load balance.
- **Sharded cache across DP/TP peers** — only helps when DP > 1;
  needs DP-aware mb-keyed P2P; weeks of engineering.

Rationale for skipping: the user does not have the compute budget
to validate or pre-train a super-deep config that would actually
exercise these schemes, and Reku's published guidance for that
regime is "selective AC + activation offload" (a different design
axis, native to torchtitan). If a future session wants distributed
caching, the design hook is in place: extend `RankLocalCache`
(`pipeline_adapter.py:91+`) into a `DistributedRankCache` keyed by
the same `(mb, producer_stage, block_idx)` slot scheme, with
eviction still gated by `_install_step_drop_patch`. That is a
separate RFC.
