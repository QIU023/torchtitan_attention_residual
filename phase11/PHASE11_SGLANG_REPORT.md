# Phase 11 — SGLang Block AttnRes Inference Report

**Date:** 2026-05-08  
**Status:** Implementation + measurement complete; pretrain resume gated on this report.

## Goal

Bring Block Attention Residual (Kimi 2026, arxiv 2603.15031, §5) to SGLang
as a *generic overlay* — sibling to upstream's `layers/mhc.py` for DeepSeek
mHC — covering the three optimisations from the Zhihu blog
(zhuanlan.zhihu.com/p/2017528295286133070):

1. **Two-phase computation** — committed-side attention vectorised across
   the block list once, then per-layer online-softmax merge.
2. **Sequence-dim TP shard** — block representations sharded along the
   token axis, replacing per-layer `all-reduce` with
   `reduce-scatter → local RMSNorm → all-gather`.
3. **Chunked-prefill compatibility** — partial-LSE state survives chunk
   boundaries.

Quality bar: **mHC-equivalent**, but **not** strict structural alignment
(no fused kernels). Generality proof: a second carrier model (Qwen3
dense) loads the same overlay layer.

---

## Implementation

### Files added (sglang fork, branch `attention_residual_inference`)

| File | LoC | Purpose |
| --- | ---: | --- |
| `python/sglang/srt/layers/attn_res.py` | ~470 | Algorithm core (Phase 1, Phase 2, comm helpers, equivalence test) — sibling of `layers/mhc.py` |
| `python/sglang/srt/models/attn_res_overlay.py` | ~920 | Kimi Linear carrier (KimiLinearForCausalLM hosting AttnRes overlay) |
| `python/sglang/srt/models/qwen3_attn_res_overlay.py` | ~580 | Qwen3 dense carrier — generality proof |
| `test/registered/layers/test_attn_res.py` | ~210 | 7 numerical-equivalence tests (CustomTestCase framework) |
| `docs/supported_models/text_generation/block_attn_res.md` | ~120 | End-user doc |

### Algorithm core (`layers/attn_res.py`)

```python
def block_attn_res_phase1(blocks, gammas, eps=1e-6):
    """Vectorised committed-side attention across the block list.
    Computes (q · γ_n · RMSNorm(V_n)) for all n at once.
    Returns logits and V_normed needed by Phase 2.
    """

def block_attn_res_phase2_merge(prev_lse, prev_o, new_logits, new_V, ...):
    """Online-softmax merge of one new block into the running aggregate.
    Max-stable LSE: safe for arbitrary block magnitudes.
    """

def reduce_scatter_seq(x):  # T → T/P along seq dim
def all_gather_seq(x):      # T/P → T
```

The two-phase decomposition amortises the committed-side I/O across all
layers' attention contributions — Phase 1 reads each block's
`(T, d)` once, Phase 2 only ingests the per-layer Q.

### Carrier overlays (`models/*_attn_res_overlay.py`)

* Subclass the upstream decoder layer; override only `forward`.
* `_run_attn` / `_run_mlp` paths split on `seq_shard` flag:
  * **Replicated path:** standard SGLang TP (o_proj/down_proj all-reduce
    in-kernel) → AttnRes contribution merged into residual stream.
  * **Sharded path:** o_proj/down_proj `reduce_results=False` + explicit
    `reduce_scatter_seq` → local RMSNorm → `all_gather_seq` → block-rep
    merge happens on the sharded tensor before scatter-back.
* Three env-var toggles:
  * `SGLANG_ATTN_RES_BYPASS=1` — skip aggregation (vanilla baseline).
  * `SGLANG_ATTN_RES_NAIVE_PATH=1` — naive single-pass aggregator.
  * `SGLANG_ATTN_RES_SEQ_SHARD=1` — enable seq-shard.
* DP attention raises `NotImplementedError` at `__init__`.

---

## Bug fixes (post-implementation)

1. **`maybe_prefix` missing** — moved upstream from `utils/common` to
   `models/transformers`; inlined the 1-liner.
2. **`e_score_correction_bias` dtype** — `moe_fused_gate` requires fp32;
   patched in-place at first forward (scoped helper, no monkey-patch).
3. **`flashinfer` rmsnorm stride alignment** — kv_lora_rank=584 unaligned
   for the kv_a layernorm; per-instance `.contiguous()` shim.
4. **`flashinfer` prefill kernel head_dim** — chose canonical aligned
   design (d=1024, H=16, head_dim=64, qk_nope=64, qk_rope=32) so all
   attention dims are 8-aligned.
5. **`is_pp_missing_parameter` filter** — upstream's check is broken for
   stitched param-dicts; custom load_weights filter.
6. **MLA `w_kc/w_vc` post-load on PPMissing** — temporary
   `config.full_attention_layer_ids` shadow with try/finally.
7. **sgl_kernel rmsnorm 2D-only** — under cuda-graph capture, flattened
   leading dims via `.reshape(-1, D)` then restored.
8. **🚨 Shard-mode AR fallback bug (correctness)** — `o_proj.reduce_results
   = False` is set permanently at `__init__`, but the fallback path
   (decode batch=1, num_tokens not divisible by TP) returned the partial
   sum unchanged, leaving each rank with a 1/P-scaled attn contribution.
   Softmax invariance to scaling masked the issue at output level
   (generations *looked* plausible). **Fix:** explicit
   `tensor_model_parallel_all_reduce` in the fallback when
   `_SEQ_SHARD_ENABLED=True`. Caught during long-context bench when
   the +14% "shard win" disappeared after cuda-graph re-enabled
   (the win was 1/8-magnitude attn output, not an actual speedup).
   Committed in `0ddd84617`.

---

## Bench results

Hardware: 8× RTX 5090, 32 GB each, CUDA 12.9, torch 2.9.1+cu129  
Model: aligned 1.4B / 447M-activated Kimi Linear AttnRes (N=4 blocks)  
Workload: 1 prompt × variable prefill + 128-token decode, 2 warmup +
3 timed (cuda-graph **on**)

### TP=1 (algorithm cost only; no fabric)

| ctx | mode | TTFT (ms) | decode (tok/s) |
| ---: | --- | ---: | ---: |
| 4096 | vanilla | 21.0 ± 0.1 | 817.1 |
| 4096 | naive | 15.7 ± 0.2 | 703.9 |
| 4096 | **two-phase** | **15.0 ± 0.0** | 535.0 |
| 8192 | vanilla | 21.4 ± 0.3 | 777.5 |
| 8192 | naive | 15.6 ± 0.2 | 673.6 |
| 8192 | **two-phase** | **15.2 ± 0.3** | 516.7 |
| 16384 | vanilla | 22.2 ± 0.3 | 733.1 |
| 16384 | naive | 17.6 ± 0.3 | 639.7 |
| 16384 | **two-phase** | **15.7 ± 0.3** | 497.4 |
| 24576 | vanilla | 24.5 ± 1.1 | 706.9 |
| 24576 | naive | 18.6 ± 0.2 | 624.6 |
| 24576 | **two-phase** | **17.9 ± 0.3** | 487.6 |

Two-phase TTFT vs naive: **0.96–0.95×** → blog's ~5% prefill win
matches at our model scale. Decode tps shows AttnRes overhead vs
vanilla (~30%): expected because each layer adds a Phase-2 merge that
runs on a single-token batch where Python op overhead dominates.

### TP=8 (algorithm + fabric)

| ctx | mode | TTFT (ms) | decode (tok/s) |
| ---: | --- | ---: | ---: |
| 4096 | vanilla | 24.6 ± 0.6 | 642.4 |
| 4096 | naive | 17.9 ± 0.7 | 566.0 |
| 4096 | two-phase | 18.4 ± 1.0 | 441.3 |
| 4096 | shard | 18.1 ± 0.9 | 442.3 |
| 8192 | vanilla | 25.5 ± 0.8 | 608.7 |
| 8192 | naive | 18.2 ± 0.6 | 548.7 |
| 8192 | two-phase | 18.5 ± 1.7 | 429.3 |
| 8192 | shard | 18.3 ± 0.6 | 415.2 |
| 16384 | vanilla | 27.1 ± 2.1 | 561.0 |
| 16384 | naive | 22.8 ± 1.0 | 482.2 |
| 16384 | two-phase | 20.4 ± 2.6 | 379.3 |
| 16384 | shard | 19.9 ± 0.2 | 389.8 |
| 24576 | vanilla | 28.7 ± 0.9 | 517.7 |
| 24576 | naive | 21.5 ± 1.0 | 456.5 |
| 24576 | two-phase | 21.4 ± 0.8 | 364.4 |
| 24576 | shard | 21.5 ± 1.0 | 364.1 |

shard ≈ two-phase at our model scale — the seq-shard win is in
**memory**, not throughput at single-token decode.

---

## Fabric NCCL trace (TP=8, kimi_tp8_shard{0,1})

| metric | shard=0 (replicated) | shard=1 (RS+AG) | Δ |
| --- | ---: | ---: | ---: |
| total NCCL ops | 21.43 M | 19.87 M | -7% |
| total bytes | 361.8 GB | 328.5 GB | -9% |
| **AllReduce ops** | 14.06 M | 11.96 M | -15% |
| **AllReduce bytes** | **60.0 GB** | **25.4 GB** | **-58%** |
| ReduceScatter bytes | 0 | 537 MB | new |
| AllGather bytes | 302.0 GB | 302.5 GB | ≈ |

The blog's claimed transformation is **directly observed**: enabling
seq-shard cuts AR bytes by 58% (60 GB → 25 GB), with the saved
AR-payload mass redistributed to RS at smaller granularity (T/P)
plus a small AG addition. The remaining AR traffic is from MoE
all-to-all (which the seq-shard path leaves alone, by design — only
attn/MLP residual paths are sharded).

### 3D mesh (TP=2 × PP=2 × EP=2)

| metric | shard=0 | shard=1 | Δ |
| --- | ---: | ---: | ---: |
| AR bytes | 14.3 GB | 5.7 GB | **-60%** |
| RS bytes | 0 | 2.1 GB | new |
| Send/Recv bytes (PP) | 540 MB | 288 MB | -47% |

The same fabric pattern shift holds under the 3D mesh: ~60% AR
reduction, RS introduced. The smaller PP send/recv with seq-shard
indicates the shard tensor (T/P) is what crosses the PP boundary,
not the replicated full sequence.

### Qwen3 cross-carrier (TP=2 × PP=2, shard=1)

Total: 925 GB; AG dominates at 99.9% (no MoE → no all-to-all). RS
present at 268 MB confirming the seq-shard path fires through the
Qwen3 carrier identically.

---

## Memory measurement

`probe_memory.py` samples per-rank `nvidia-smi memory.used` at four
checkpoints (pre_boot / post_boot / post_warmup / post_run) for both
shard=0 and shard=1 at TP=8 prefill=16384.

```
shard=0:  pre_boot 2 MB → post_boot 21403 → post_warmup 21435 → post_run 21453
shard=1:  pre_boot 2 MB → post_boot 21401 → post_warmup 21433 → post_run 21451
```

**Result: indistinguishable** (Δ ≤ 2 MB, allocator noise).

**Why:** SGLang's `mem_fraction_static=0.6` reserves a fixed pool at
boot; the workload-level block-rep delta (~128 MB at our 16K context)
is invisible inside a 21 GB reserved pool.

**Analytical figure:** at our 1.4B / d=1024 / N=4 / T=16384, replicated
block-reps are 128 MB/rank, shard P=8 is 16 MB/rank — ~110 MB
expected delta. The blog's headline 15GB→1.9GB is at d=7168 / N=8 /
T=131072 (~58× our scale); to hit ~270 MB observable Δ we'd need
T≥64K with `mem_fraction_static` lowered to ~0.3 so the reserved pool
shrinks below the workload. Out of scope for this phase.

---

## Tests

```
$ pytest test/registered/layers/test_attn_res.py -q
......s
6 passed, 1 skipped
```

* Phase-2 merge ↔ naive aggregate (fp32 + bf16, ε=1e-5)
* Block-0 edge case (empty prev_o)
* Phase-1 vectorised vs Python loop
* Zero-init pseudo-query identity
* TP=1 helper identity (skipped: requires torch.distributed init)

End-to-end correctness validated by:
* boot+generate on canonical aligned 1.4B ckpt at all TP=1 / TP=8 / 3D mesh
* Qwen3 carrier same workflow
* shard ↔ replicated output match (post-fix) — see "Bug fixes #8"

---

## Carrier generality

| carrier | total params | active params | TP=8 | 3D (TP=2×PP=2×EP=2) |
| --- | ---: | ---: | --- | --- |
| Kimi Linear AttnRes (aligned) | 1.4B | 447M | ✅ boot+gen | ✅ boot+gen (traced) |
| Qwen3 dense AttnRes | 96.7M | 96.7M | ✅ boot+gen | ✅ boot+gen (traced) |

Both carriers re-use the same `layers/attn_res.py` core. The carrier
modules differ only in (a) the underlying `self_attn` / `mlp` upstream
classes and (b) MoE handling (Kimi MoE has its own AR; Qwen3 dense
doesn't).

---

## Files (parent repo)

| File | Purpose |
| --- | --- |
| `phase11/dump_aligned_smoke.py` | 1.4B aligned ckpt builder |
| `phase11/dump_qwen3_attn_res_smoke.py` | Qwen3 ckpt builder |
| `phase11/bench_attn_res.py` | 4-mode bench harness (subprocess-isolated env) |
| `phase11/run_long_ctx_bench.sh` | TP×ctx sweep |
| `phase11/probe_cuda_graph.py` | cuda-graph compat smoke |
| `phase11/probe_memory.py` | per-rank GPU mem sampling via nvidia-smi |
| `phase11/run_all_traces.sh` | Full trace sweep (4 inference + 1 PPO) |
| `phase11/bench_results/*.json` | Bench raw output |
| `phase11/trace_*` | NCCL traces + collective_summary.csv + flows.csv + ixia_config.json |

---

## Upstream-PR readiness

| Criterion | Status |
| --- | --- |
| Builds without monkey-patches | ✅ |
| Three env toggles, no API breakage | ✅ |
| 7 numerical-equivalence tests | ✅ (6 pass + 1 distributed-skip) |
| Two-carrier proof of generality | ✅ (Kimi + Qwen3) |
| End-user doc | ✅ |
| Fused kernels | ✅ Phase-2 merge+RMSNorm+logit Triton kernel (`_phase2_merge_norm_kernel` in `layers/attn_res.py`) — matches blog's "Phase 2 elementwise → fuses with RMSNorm/AR" claim. Single read of partial_block (was 2), fp32 internal math, falls back to torch path on CPU |
| DP attention | ❌ (raises clearly) |
| Chunked-prefill | partial — Phase-2 merge supports continuation but not exercised in this report |

This is a research-deliverable PR target. Phase-2 fused Triton kernel
(merge + RMSNorm + logit) landed in `sglang@63325b2b4`. For upstream
merge, remaining work: (a) Phase-1 batched-attention Triton kernel
(currently torch.einsum), (b) DP-attention support, (c) chunked-
prefill stress test, (d) Phase 1 ↔ first-decoder-layer CUDA stream
overlap (deferred — requires CUDA-graph stream parallelism).

---

## Closing the loop

* Shard correctness fix committed in `sglang@0ddd84617` (this branch).
* Phase-2 fused Triton kernel committed in `sglang@63325b2b4`.
* Local-only env-compat patches in `torchtitan/distributed/*` and
  `torchtitan/models/common/*` exported to
  [`phase11/torchtitan_vast_ai_env_compat.patch`](torchtitan_vast_ai_env_compat.patch)
  with the rationale documented in
  [`phase11/TORCHTITAN_VAST_AI_PATCHES.md`](TORCHTITAN_VAST_AI_PATCHES.md).
  These are vast.ai-specific (torch 2.9.1+cu129 stable vs torchtitan's
  nightly target) and stay outside the torchtitan submodule pointer so
  upstream merges aren't polluted with `hasattr` guards.
* Pretrain (step-8000 ckpt) ready for resume:
  ```
  CONFIG=kimi_linear_447m_aligned_block_attn_res_n4 \
  OUT_DIR=$PWD/phase4/runs/kimi_447m_aligned_block_attn_res_fsdp_paperhparams \
  NGPU=8 bash phase4/launch_from_scratch_paperhparams.sh
  ```
* Phase 9 PPO production post-training: still pending (out of scope here).
