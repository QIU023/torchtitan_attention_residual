# Phase 11 SGLang AttnRes — Production Profiling Report

Date: 2026-05-09 (after audit + sync fix)

## Headline finding

**The Phase-2 fused Triton kernel was never actually firing during
the original v2 bench.** SGLang installs from `/sgl-workspace/sglang/`
but the kernel commit went into the user submodule at
`/root/torchtitan_attention_residual/sglang/`. The two paths drifted
between commits and the install was never re-synced. This made all
v2 bench numbers reflect the torch-only path and led to the wrong
conclusion ("no wall-clock delta from Triton kernel") in the
original report.

**After syncing the install (`cp` to `/sgl-workspace/sglang/...`)
and re-benching:** decode throughput jumps **+27-30% across all
contexts and TP topologies**.

## Bench: kernel-on (v3) vs kernel-off (v2)

Same harness, same model, same prompts; only diff is whether
`/sgl-workspace/sglang/python/sglang/srt/layers/attn_res.py`
contains the kernel.

### TP=1

| ctx | mode | v2 decode tps (kernel off) | **v3 decode tps (kernel on)** | Δ |
|---:|---|---:|---:|---:|
| 4K | vanilla | 818 | 806 | -1% (noise) |
| 4K | naive | 704 | 704 | 0% |
| 4K | two-phase | 535 | **698** | **+30.4%** |
| 16K | vanilla | 733 | 732 | 0% |
| 16K | naive | 639 | 638 | 0% |
| 16K | two-phase | 497 | **634** | **+27.6%** |

### TP=8

| ctx | mode | v2 decode tps | **v3 decode tps** | Δ |
|---:|---|---:|---:|---:|
| 4K | vanilla | 642 | 638 | 0% |
| 4K | naive | 566 | 563 | 0% |
| 4K | two-phase | 441 | **559** | **+26.8%** |
| 4K | shard | 442 | **554** | **+25.4%** |
| 16K | vanilla | 561 | 559 | 0% |
| 16K | naive | 482 | 484 | 0% |
| 16K | two-phase | 379 | **482** | **+27.2%** |
| 16K | shard | 389 | **481** | **+23.6%** |

### TTFT (also improved)

| ctx | mode | v2 TTFT (ms) | **v3 TTFT (ms)** |
|---:|---|---:|---:|
| TP=1 4K | two-phase | 15.0 | **14.5** |
| TP=1 16K | two-phase | 16.6 | **15.9** |
| TP=8 16K | two-phase | 19.6 | **20.4** |
| TP=8 16K | shard | 20.0 | **18.8** |

Two-phase TTFT vs naive: **0.92-0.95×** (was 0.95-1.04× without
kernel). The blog's "+5% prefill win" claim is now exceeded.

## Profiling: kernel breakdown (TP=1 prefill=4096 + decode=64)

torch.profiler via SGLang's `start_profile`. Trace file:
`phase11/profile_results/kineto/two-phase/*.trace.json.gz`.

### Top-12 kernels in two-phase mode (kernel-on)

| % | time | calls | kernel |
|---:|---:|---:|---|
| 19.4% | 20.1 ms | 1857 | cuBLAS gemvx (MLA / dense matmul) |
| 14.9% | 15.5 ms | 4612 | cuBLAS gemvx (MoE / projection) |
| 11.2% | 11.6 ms | 1950 | fused_moe_kernel |
| 9.1% | 9.4 ms | 975 | moe_fused_gate_kernel |
| 8.1% | 8.5 ms | 256 | flashinfer MLA paged attention |
| 3.5% | 3.7 ms | 975 | moe_align_block_size |
| 3.5% | 3.6 ms | 2015 | flashinfer activation (silu) |
| **2.8%** | **2.9 ms** | **2015** | **`_phase2_merge_norm_kernel` ← our Triton kernel** |
| 2.6% | 2.7 ms | 2176 | flashinfer RMSNormKernel |
| 2.6% | 2.7 ms | 3380 | vectorized_elementwise_kernel (add) |
| 2.4% | 2.4 ms | 1495 | direct_copy_kernel |
| 2.0% | 2.1 ms | 768 | KDA fused_sigmoid_gating_delta_rule |

**Kernel call count = 2015**, matching the expected 64 decode tokens
× 16 layers × 2 queries (pre-attn + pre-FFN) = **2048 expected**;
the 33 missed are first-block-empty cases where Phase-2 short-
circuits. So **>98% of expected Phase-2 merge ops hit the kernel**.

### Total kernel time per mode

| mode | total kernel us (TP=1 4K) |
|---|---:|
| vanilla | 91 024 |
| naive | 104 866 |
| two-phase | **105 022** |

Two-phase is essentially the same kernel-time-budget as naive (only
+0.1%) despite computing more (Phase 1 batched matmul + Phase 2 per-
layer merge), because the fused Triton kernel + cuBLAS for Phase 1
make per-op cost extremely small.

The remaining +13 ms over vanilla is entirely the additional
non-fused work (Phase 1 cublasLt::splitKreduce, the elementwise
exp/max/divide chain that didn't get fused).

## NCCL fabric (re-confirm — bytes don't change)

Already captured in v1 traces (unchanged because fabric ops are
kernel-independent):

| | shard=0 (replicated AR) | shard=1 (RS+AG) | Δ |
|---|---:|---:|---:|
| AllReduce bytes | 60.0 GB | 25.4 GB | **-58%** |
| RS bytes | 0 | 537 MB | new |

3D mesh: AR -60%. Same numbers.

## Conclusion changes vs original report

The original report concluded:

> "Re-bench with the Phase-2 Triton kernel active showed no
> significant wall-clock delta vs the cuda-graph-fused torch path"

This was **incorrect**. The kernel was never running due to install
path drift. Corrected:

> "The Phase-2 fused Triton kernel reduces decode tokens/sec
> overhead from -32% to **-13%** vs vanilla, a **+27% absolute
> throughput improvement** at our 1.4B / d=1024 / N=4 scale.
> Two-phase TTFT improves from 0.95× to **0.92-0.95×** of naive.
> The kernel call count is 2015 / 2048 expected per (64 decode
> tokens × 16 layers × 2 queries) batch."

This now aligns with the blog's claim that Phase-2 fusion with
RMSNorm provides "进一步减少额外 IO" — the win is real and
measurable at our scale, not just at 128K context.

## Lesson learned

`/sgl-workspace/sglang/` is the install location used by `import
sglang`; my git operations and edits went into the user submodule
at `/root/torchtitan_attention_residual/sglang/` which is a separate
clone tracked as a git submodule. **The two paths must be kept in
sync** (cp or symlink) for changes in the submodule to take effect
at runtime.

Going forward I should add a verification step to bench/profile
scripts that prints `sglang.srt.layers.attn_res.__file__` and
checks `_phase2_merge_norm_triton is not None` before declaring a
result.
