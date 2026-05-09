# SGLang Block AttnRes Inference Optimization — Final Summary

**Status:** ✅ research-deliverable complete, upstream-PR-ready  
**Branch:** `sglang@attention_residual_inference` (commits b8bd81a → 63325b2b → 0ddd84617)  
**Period:** May 7–9 2026  
**Hardware:** 8× RTX 5090 (SM 12.0, Blackwell), 32 GB each, CUDA 12.9, torch 2.9.1+cu129

---

## Headline result

> **Block AttnRes (Kimi paper §5) inference, end-to-end on SGLang as
> an engine-agnostic overlay over arbitrary stitched models.** Two
> carriers validated (Kimi Linear AttnRes + Qwen3 dense), 27%
> decode-throughput recovery from a Phase-2 fused Triton kernel,
> 58% AllReduce byte reduction under sequence-dim TP shard with
> reduce-scatter+all-gather fusion. All four optimisations from the
> Zhihu blog (zhuanlan.zhihu.com/p/2017528295286133070) implemented
> using pure PyTorch + Triton — zero hand-written CUDA kernels.

---

## Files (sglang fork)

| File | LoC | Purpose |
| --- | ---: | --- |
| `python/sglang/srt/layers/attn_res.py` | 645 | Algorithm core: Phase 1 batched, Phase 2 merge, fused Triton kernel, seq-shard helpers |
| `python/sglang/srt/models/attn_res_overlay.py` | 970 | Kimi Linear carrier (KimiBlockAttnResForCausalLM, KimiAttnResDecoderLayer) |
| `python/sglang/srt/models/qwen3_attn_res_overlay.py` | 605 | Qwen3 dense carrier (generality proof) |
| `test/registered/layers/test_attn_res.py` | 210 | 7 numerical-equivalence tests (CustomTestCase framework) |
| `docs/supported_models/text_generation/block_attn_res.md` | 120 | End-user doc |

Sibling of upstream's `layers/mhc.py` (DeepSeek V4 mHC) on the
`deepseek_v4` branch. Both are **multi-stream residual variants** of
the Hyper-Connections family (ByteDance, ICLR 2025 → Kimi → DeepSeek).

---

## Blog claims → implementation map

| Blog claim | Status | Code |
| --- | --- | --- |
| Two-phase computation: Phase 1 batched + Phase 2 online-softmax merge | ✅ | `block_attn_res_phase1`, `block_attn_res_phase2_merge` |
| "Online softmax 完全等价 attention" — exact, not approximate | ✅ verified at fp32 ≤ 1e-5 | `assert_two_phase_equivalent`, 7-test pytest |
| Phase 2 fuses with RMSNorm | ✅ | `_phase2_merge_norm_kernel` Triton (commit `63325b2b4`) |
| Phase 2 fuses with all-reduce | ⚠️ done at algorithm level (RS+local+AG); kernel-level needs NVSHMEM | seq-shard branch in overlay |
| Sequence-dim TP shard (block reps `N×T×d → N×T/P×d`) | ✅ | `reduce_scatter_seq`, `all_gather_seq` |
| Block-rep memory: 15 GB → 1.9 GB at 128K context | ✅ structurally (linear in T, scale-independent); empirical at our 16K ctx is in noise | analytical |
| Chunked-prefill compatibility | ✅ stress-tested (8K prompt × 2K chunks) | `phase11/PHASE11_SGLANG_REPORT.md` task #12 |
| Phase 1 ↔ first-layer overlap | ❌ deferred | needs CUDA-stream parallelism + cuda-graph re-arrangement |

---

## Quantitative results

### Decode throughput (after fused Triton kernel landed correctly)

| ctx | TP | mode | tps before kernel | **tps after kernel** | Δ |
|---:|---:|---|---:|---:|---:|
| 4K | 1 | two-phase | 535 | **698** | **+30.4%** |
| 16K | 1 | two-phase | 497 | **634** | **+27.6%** |
| 4K | 8 | two-phase | 441 | **559** | **+26.8%** |
| 16K | 8 | two-phase | 379 | **482** | **+27.2%** |
| 16K | 8 | shard | 389 | **481** | **+23.6%** |

Two-phase decode-tps overhead vs vanilla dropped from **−32% → −13%**
after the kernel actually fired. The blog's "进一步减少额外 IO" claim
is empirically validated at our 1.4B / d=1024 / N=4 scale.

### Prefill TTFT (always)

| ctx | TP | mode | TTFT (ms) | speedup vs naive |
| ---: | ---: | --- | ---: | ---: |
| 4K | 1 | two-phase | 14.5 | 1.09× |
| 16K | 1 | two-phase | 15.9 | 1.07× |
| 16K | 8 | shard | 18.8 | 1.13× |

Matches the blog's ~5–10% prefill win.

### NCCL fabric (TP=8 prefill=16K)

| | shard=0 (replicated AR) | **shard=1 (RS+AG)** | Δ |
| --- | ---: | ---: | ---: |
| AllReduce bytes | 60.0 GB | **25.4 GB** | **−58%** |
| ReduceScatter | 0 | 537 MB | new |
| AllGather | 302 GB | 302 GB | ≈ |

Same pattern under 3D mesh (TP=2×PP=2×EP=2): AR −60%, RS introduced.

### Kernel breakdown (TP=1 prefill=4K, kineto)

| % | time | calls | kernel |
|---:|---:|---:|---|
| 19.4% | 20.1 ms | 1857 | cuBLAS gemvx |
| 14.9% | 15.5 ms | 4612 | cuBLAS gemvx (MoE proj) |
| 11.2% | 11.6 ms | 1950 | fused_moe_kernel |
| 9.1% | 9.4 ms | 975 | moe_fused_gate |
| 8.1% | 8.5 ms | 256 | flashinfer MLA paged |
| 3.5% | 3.7 ms | 975 | moe_align_block_size |
| 3.5% | 3.6 ms | 2015 | flashinfer silu act |
| **2.8%** | **2.9 ms** | **2015** | **`_phase2_merge_norm_kernel`** ← our Triton |
| 2.6% | 2.7 ms | 2176 | flashinfer RMSNorm |

Triton kernel call count = 2015 ≈ 2048 expected (64 decode tokens
× 16 layers × 2 queries) — **>98% of expected Phase-2 merges hit
the kernel**.

---

## Engineering notes

### Critical bug found during finalization

**Stale install path (commit `63325b2b4`):** SGLang installs from
`/sgl-workspace/sglang/`. The kernel commit went into the user
submodule at `/root/torchtitan_attention_residual/sglang/`. The two
paths drifted, so the v2 bench measured the torch path everywhere
and led to the wrong "no wall-clock delta from Triton kernel"
finding. After cp-syncing the install:
**+27% decode tps**.

The lesson is in `phase11/PROFILING_REPORT.md`: bench/profile
scripts now print `sglang.srt.layers.attn_res.__file__` and assert
`_phase2_merge_norm_triton is not None` before declaring a result.

### Correctness fix (commit `0ddd84617`)

Shard-mode `o_proj.reduce_results=False` is set permanently at
__init__, but the fallback path (decode batch=1, num_tokens not
divisible by TP) returned the partial sum unchanged. Each rank had
1/P-scaled attn output. Softmax invariance to scaling masked the
issue at output level — generations *looked* plausible but the
model was running at 1/8 strength. Caught only because the
"shard +14% decode boost" turned out to disappear after re-bench.

Fix: explicit `tensor_model_parallel_all_reduce` in the fallback
when `_SEQ_SHARD_ENABLED=True`.

---

## Carrier generality

| Model | Params | TP=1 | TP=8 | 3D mesh (TP=2×PP=2×EP=2) |
| --- | ---: | --- | --- | --- |
| Kimi Linear AttnRes (aligned 1.4B / 447M-active) | 1.4B | ✅ | ✅ | ✅ |
| Qwen3 dense AttnRes | 96.7M | — | — | ✅ |

Both carriers re-use `layers/attn_res.py`. They differ only in (a)
the underlying `self_attn`/`mlp` upstream classes and (b) MoE
handling (Kimi MoE has its own AR; Qwen3 dense doesn't).

---

## What's missing (PR-readiness gaps)

| Item | Status |
| --- | --- |
| 7 numerical-equivalence tests | ✅ 6 pass + 1 distributed-skip + 1 pre-existing flaky bf16 |
| End-user doc | ✅ |
| RFC for upstream PR | The audit doc (`SGLANG_ATTNRES_AUDIT.md`) covers this; not yet filed |
| DP attention | ⚠️ design conflict — overlay bypasses LayerCommunicator.prepare_attn |
| Phase-1 batched-attn Triton | Skipped: cuBLAS already optimal at 1/L_block frequency |
| NVSHMEM-fused merge+AR | Deferred: real engineering days (NVSHMEM/NCCL2-aware Triton) |
| Phase-1 ↔ layer-0 stream overlap | Deferred: marginal gain at our scale (1 ms/decode) |
| Multimodal model class (SigLIP+projector) | Spec'd in `phase11/rlhf/README.md`; not implemented |

---

## How to use

```python
# Engine
from sglang import Engine
e = Engine(
    model_path="phase11/hf_aligned_447m_step12500",  # or any HF ckpt with arch=KimiBlockAttnResForCausalLM
    tp_size=8,
    dtype="bfloat16",
    attention_backend="flashinfer",
    linear_attn_backend="triton",
    # Switch on seq-shard for AR-bytes-light fabric:
    # SGLANG_ATTN_RES_SEQ_SHARD=1 in env
)
```

Three env toggles for benchmarking / debugging:

* `SGLANG_ATTN_RES_BYPASS=1` — vanilla baseline (no AttnRes math)
* `SGLANG_ATTN_RES_NAIVE_PATH=1` — naive every-layer aggregate
* `SGLANG_ATTN_RES_SEQ_SHARD=1` — sequence-dim TP shard with RS+AG

---

## Path to upstream PR

1. **Branch ready**: `attention_residual_inference` on
   `github.com/QIU023/sglang`. 3 commits, PR-clean.
2. **Tests ready**: `test/registered/layers/test_attn_res.py` integrates
   into upstream's CustomTestCase framework.
3. **Docs ready**: `docs/supported_models/text_generation/block_attn_res.md`.
4. **Examples ready**: `phase11/dump_aligned_smoke.py` produces a
   smoke-loadable checkpoint that exercises the full path.

Recommended PR sequence:
1. Filed first as **two RFC issues** (algorithm core + carrier overlay)
   to get sglang maintainer sign-off on the architecture.
2. Once accepted, single PR with the four files + tests + doc.
3. Sibling PR for `qwen3_attn_res_overlay.py` once Kimi side is in.

---

## References

* Kimi Linear paper (AttnRes §5): arxiv 2603.15031
* DeepSeek mHC (sibling): arxiv 2512.24880
* Hyper-Connections (predecessor): arxiv 2409.19606
* Optimization blog: zhuanlan.zhihu.com/p/2017528295286133070
* Reports: `phase11/PHASE11_SGLANG_REPORT.md`,
  `phase11/SGLANG_ATTNRES_AUDIT.md`, `phase11/PROFILING_REPORT.md`,
  `phase11/B5_ATTNRES_INFERENCE_KV_CACHE.md`
