# KDA Blackwell sm_120 Optimization Analysis

**Date:** 2026-05-03  
**Context:** RTX 5090 (Blackwell GB202, sm_120), observed MFU ≈ 0.21%  
**Scope:** Research/documentation only — no upstream fla-core code changes proposed

---

## Section 1 — Current chunk_kda Kernel Structure

### 1.1 Kernel Dispatch Graph

`chunk_kda` is not a single monolithic kernel. It dispatches to a pipeline of six sub-operations:

```
chunk_kda_fwd()
├── kda_gate_chunk_cumsum()          [gate.py: kda_gate_chunk_cumsum_vector_kernel]
├── chunk_kda_fwd_intra()            [chunk_intra.py: chunk_kda_fwd_kernel_intra_sub_chunk
│                                                  + chunk_kda_fwd_kernel_inter_solve_fused]
├── chunk_gated_delta_rule_fwd_h()   [common/chunk_delta_h.py: chunk_gated_delta_rule_fwd_kernel_h_blockdim64]
└── chunk_gla_fwd_o_gk()             [gla/chunk.py: chunk_gla_fwd_kernel_o]
```

Each sub-kernel has independent autotune configs. The backward pass adds:
```
chunk_kda_bwd()
├── chunk_kda_bwd_kernel_dAv         [chunk_bwd.py]
└── chunk_kda_bwd_kernel_wy_dqkg_fused [chunk_bwd.py]
```

### 1.2 Block Sizes (from autotune configs)

| Kernel | BT (chunk) | BK | BV | num_warps | num_stages |
|--------|-----------|-----|-----|-----------|------------|
| `kda_gate_chunk_cumsum_vector_kernel` | 16–32 or 32–64 | — | — | 2, 4, 8 | 2, 3 |
| `kda_gate_fwd_kernel` | 32, 64, 128 | — | — | 4, 8, 16, 32 | 2, 3 |
| `chunk_kda_fwd_kernel_intra_sub_chunk` | BT (outer) | BC (sub-chunk) | — | 1, 2, 4, 8 | 2, 3, 4 |
| `chunk_kda_fwd_kernel_inter_solve_fused` | — | 32, 64 | — | 1, 2, 4 | (default) |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 64 (fixed) | — | 32–64 | 2, 4 | 1–4 |
| `chunk_gla_fwd_kernel_o` | BT (outer) | 32, 64 | 64, 128 | 2, 4, 8 | 2, 3, 4 |
| `chunk_kda_bwd_kernel_wy_dqkg_fused` | BT (outer) | 16–64 | 16–128 | 2, 4 | 2, 3, 4 |

**Top-level chunk size** `BT = 64` is hardcoded in `chunk.py` (the Python wrapper).

### 1.3 Architecture-Awareness

The kernels implement two tiers of architecture detection:

**Tier 1 — `check_shared_mem()` guards (used in bwd kernel `chunk_kda_bwd_kernel_wy_dqkg_fused`):**
- `check_shared_mem()` → allows `BK ∈ {32, 64}` (vs `{16, 32}` on older GPUs)
- `check_shared_mem('ampere')` → allows `BV ∈ {64, 128}` (vs `{16, 32}`)
- `check_shared_mem('ada')` → allows `BV ∈ {32, 64}` in the h-kernel

**Tier 2 — `IS_NVIDIA_HOPPER` guards (used in bwd kernels):**
- On Hopper: `num_warps ∈ [2, 4]` only (avoids warp oversubscription on large SM)
- On non-Hopper: `num_warps ∈ [2, 4, 8]`
- Hopper-specific config exclusion: `not (IS_NVIDIA_HOPPER and BK == 32 and num_warps == 4)`

**Critical gap:** There is no `IS_NVIDIA_BLACKWELL` detection anywhere. The codebase has three architecture tiers: legacy (≤ Turing), Ampere, Ada/Hopper. Blackwell falls into the Hopper branch (or possibly Ada) based on `check_shared_mem()` thresholds — but this has not been verified and the Blackwell-specific register/shared-memory increases are not exploited.

### 1.4 MMA Instruction Style

- All kernels use standard `tl.dot()` Triton operations
- **No `wgmma.mma_async`** (Hopper Warp Group MMA) is used explicitly
- **No TMA (Tensor Memory Accelerator)** descriptors are used
- Triton's backend chooses the lowered PTX/SASS instruction; on Hopper sm_90 this typically lowers to `wgmma` when `tl.dot()` is used with appropriate tile sizes. On Blackwell sm_120, Triton 3.x would lower to `tcgen05.mma` automatically for qualifying tile sizes — but only if Triton's sm_120 backend support is complete (as of early 2025 this was still being finalized).

### 1.5 Shared Memory Estimate

For the main output kernel `chunk_gla_fwd_kernel_o` with BK=64, BV=128, BF16:
- Q tile: BT × BK × 2 bytes = 64 × 64 × 2 = 8 KB
- V tile: BT × BV × 2 bytes = 64 × 128 × 2 = 16 KB
- Hidden state h: BK × BV × 4 bytes = 64 × 128 × 4 = 32 KB (fp32)
- Pipeline buffer (num_stages=4): × 4 × (K+V tiles) ≈ additional 96 KB
- **Estimated peak: ~60–100 KB** per block under typical autotune configs

For the h-kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` with BV=64:
- State matrix: 64 × 64 × 4 bytes = 16 KB
- W matrix (BT × BK): 64 × 64 × 2 = 8 KB  
- **Estimated peak: ~30–50 KB** per block

### 1.6 Register Pressure Estimate

- Gate kernel (BT=128, num_warps=32): high register pressure, ~64–96 registers/thread estimated
- h-kernel (BT=64, BV=64, num_warps=4): moderate, ~48–64 registers/thread estimated
- Output kernel (BK=64, BV=128, num_stages=4): high due to pipelining, ~80–100 registers/thread estimated

Register pressure is a known bottleneck for num_stages > 3 on these tile sizes.

---

## Section 2 — Blackwell sm_120 Architecture Deltas vs Hopper sm_90

### 2.1 Die-Level Specs (GB202, RTX 5090)

| Specification | Hopper H100 (SXM5) | Blackwell GB202 (RTX 5090) |
|--------------|---------------------|---------------------------|
| SMs | 132 | 192 |
| CUDA cores/SM | 128 | 128 |
| Total CUDA cores | 16,896 | 24,576 |
| Tensor cores (5th gen) | 4th gen | 5th gen (MXFP8, MXFP6, MXFP4) |
| L1 / shared mem per SM | 228 KB (configurable) | 128 KB (L1 fixed, shared configurable; total ~228 KB) |
| L2 cache (total) | 50 MB (H100 SXM) | 128 MB (GB202) |
| GDDR/HBM | 80 GB HBM3 (H100) | 32 GB GDDR7 |
| Memory bandwidth | 3.35 TB/s (H100 SXM) | ~1.79 TB/s (RTX 5090, GDDR7 512-bit) |
| NVLink | NVLink 4.0 (900 GB/s) | None on consumer card |
| FP16 Tensor (no sparsity) | 989 TFLOPS | ~838 TFLOPS (estimated from 3,352 AI TOPS FP8) |
| FP8 Tensor | ~1,979 TFLOPS | ~3,352 TOPS |
| FP4 Tensor | N/A | ~6,704 TOPS (estimated) |

**Note on shared memory per SM:** The Wikipedia entry reports 128 KB L1 for GB202. NVIDIA's documentation for server Blackwell (GB100/GB200) lists 232 KB shared memory per SM. For the consumer GB202, the shared memory per SM is likely similar to Ada Lovelace (~100 KB usable as shared mem, up to 228 KB combined L1+shared) but this has not been confirmed by NVIDIA's official RTX 5090 datasheet. The `check_shared_mem('ada')` path in fla-core would apply.

### 2.2 New Blackwell Instructions Relevant to Triton Kernels

**tcgen05.mma (Blackwell Tensor Core MMA):**
- Blackwell introduces `tcgen05.mma` PTX instructions, the successor to Hopper's `wgmma.mma_async`
- `tcgen05` uses **Tensor Memory** (TMEM), a new per-SM scratchpad distinct from shared memory
- TMEM: ~512 KB per SM on server Blackwell (GB100); consumer GB202 size unconfirmed
- Fragment shapes: supports M=64, N=8..256, K=16 for FP16; M=128 for FP8
- Key difference from `wgmma`: accumulator lives in TMEM (not registers), reducing register pressure
- Triton 3.3+ (≥ early 2025) adds initial sm_120 support; full `tcgen05` lowering was in progress as of the knowledge cutoff

**DSMEM (Distributed Shared Memory):**
- Blackwell enables direct peer-SM shared memory reads without going through L2
- Allows a warp to load data from a neighbor SM's shared memory
- Useful for "distributed softmax" and "distributed tiling" patterns
- Not directly accessible from Triton 3.x (requires inline PTX or CuTe/CUTE library)
- Relevant for: inter-chunk state passing in the h-kernel (currently goes through global memory)

**TMA (Tensor Memory Accelerator):**
- TMA was introduced in Hopper sm_90 and is available on Blackwell sm_120 as well
- Blackwell adds **TMA multicasting** — a single TMA descriptor can broadcast to multiple SMs
- Useful for parameter tiles that are reused across heads (e.g., the hidden state h is read by multiple output blocks)
- Triton supports TMA via `tl.make_block_ptr()` + `tl.load()` with boundary checks (experimental in Triton 3.x)

**FP8 / MX Formats:**
- Blackwell 5th-gen Tensor Cores natively support OCP MXFP8, MXFP6, MXFP4
- MXFP8 = per-block scaling (block size 32 elements), different from H100's E4M3/E5M2 FP8
- For KDA: q, k are L2-normalized in-kernel (`use_qk_l2norm_in_kernel=True`), which means dynamic range is bounded — suitable for FP8 quantization

**Register File:**
- Blackwell maintains 65,536 × 32-bit registers per SM (same as Hopper)
- But `tcgen05` moves MMA accumulators to TMEM, freeing ~50% of the register pressure from `wgmma` accumulators — this is a major difference

### 2.3 Triton sm_120 Readiness (as of 2025)

Triton's sm_120 support progression:
- Triton 3.1 (late 2024): basic sm_120 compilation target added
- Triton 3.2 (early 2025): tcgen05 backend in progress; `tl.dot()` on sm_120 may still lower to legacy `mma` in some configurations
- Full tcgen05 + TMEM support was not complete as of early 2025
- **Practical implication:** fla-core running on sm_120 today may not benefit from tcgen05 even if block sizes are Blackwell-optimal, because Triton's backend may fall back to older instruction sequences

---

## Section 3 — Concrete Optimization Opportunities

### OPP-1: Add `IS_NVIDIA_BLACKWELL` detection + dedicated autotune configs

**What changes:**  
Add `IS_NVIDIA_BLACKWELL = torch.cuda.get_device_capability() >= (12, 0)` alongside the existing `IS_NVIDIA_HOPPER` check. Expand autotune grids for Blackwell: `num_warps ∈ [4, 8, 16]` (Blackwell SM has 4 warp schedulers like Hopper but with wider TMEM), `num_stages ∈ [3, 4, 5]`.

**Why it helps Blackwell:**  
Current Hopper branch limits `num_warps ∈ [2, 4]` in the backward kernels. On Blackwell, 8 warps per block can better hide TMEM latency. The `num_stages=5` pipeline can exploit Blackwell's deeper async pipeline without extra shared mem cost because TMEM holds accumulators.

**Rough speedup estimate:** 10–20% kernel time on the affected kernels (bwd kernels primarily). Lower bound because autotuner would find the right config; without this, it runs suboptimal Hopper configs.

**Difficulty:** triton-easy — pure autotune config change, no kernel body edits.

---

### OPP-2: Increase BT from 64 to 128 (hardcoded chunk size)

**What changes:**  
The top-level chunk wrapper (`chunk.py`) hardcodes `chunk_size = 64`. This sets BT for all sub-kernels. Increasing to 128 doubles the tile area processed per kernel launch.

**Why it helps Blackwell:**  
- Fewer kernel launches per sequence → less launch overhead (Blackwell has higher per-launch overhead than Hopper due to no persistent warp support in Triton)
- Larger BT increases arithmetic intensity of the intra-chunk kernel (O(BT²·K) ops vs O(BT·K) loads) → better compute/memory ratio
- Blackwell's 128 MB L2 (vs 50 MB on H100) can hold larger working sets, reducing reload pressure for multi-head batches
- L2 hit rate improves because K/V chunks of size 128 × K fit in L2 across adjacent blocks

**Tradeoff:** BT=128 doubles SRAM for intra-chunk lower-triangular attention (BT² × 2 bytes = 32 KB vs 8 KB for BT=64). Must verify it fits within shared mem budget.

**Rough speedup estimate:** 15–30% for the intra-chunk kernel which is O(BT²). Less impact on the h-kernel (O(BT·K·V)) which is already compute-bound.

**Difficulty:** triton-easy to medium — change one constant, re-run autotune, verify numerics unchanged.

---

### OPP-3: BV=128 or BV=256 for the h-kernel and output kernel

**What changes:**  
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` currently caps `BV ∈ {32, 64}` gated by `check_shared_mem('ada')`. The output kernel `chunk_gla_fwd_kernel_o` allows `BV ∈ {64, 128}` on Ampere+. Adding a Blackwell tier that allows `BV=128` for the h-kernel and `BV=256` for the output kernel (if shared mem permits) would increase compute density per block.

**Why it helps Blackwell:**  
- The hidden state matrix h has shape [B, H, K, V]. For typical KDA config (K=128, V=128), the h tile at BK=64, BV=128 is 64×128×4 = 32 KB — fits in Blackwell shared mem
- Wider BV means more matmul reuse of the loaded K tile: each K element is multiplied against BV output columns. Doubling BV doubles arithmetic intensity without extra K loads
- Blackwell's 5th-gen Tensor Cores prefer larger N-dimension tiles (N=128–256 for peak efficiency with tcgen05.mma M=64 tiles)

**Rough speedup estimate:** 20–35% for h-kernel and output kernel, which together constitute the majority of chunk_kda wall time.

**Difficulty:** triton-medium — requires verifying shared memory budget per SM for the chosen (BV, BK, num_stages) combination, and adding Blackwell-specific config entries.

---

### OPP-4: Persistent kernel design for the h-kernel

**What changes:**  
Currently `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` is launched with grid `(NT, B*HV)` where NT = T/BT. Each block processes one chunk independently, then exits. A persistent kernel would keep blocks alive across chunks, passing state via registers/TMEM instead of writing to global memory.

**Why it helps Blackwell:**  
- State `h` (shape [B, H, K, V] = e.g. 1×16×128×128×4 = 8 MB per layer) is written to global memory after each chunk and re-read by the output kernel. With persistent warps, `h` at chunk boundary stays in TMEM/registers
- Eliminates one global memory round-trip per chunk: at BT=64 and seqlen=2048, that's 32 round-trips of 8 MB = 256 MB of unnecessary traffic per layer per forward pass
- Blackwell's TMEM (512 KB/SM on server; ~256 KB estimated on consumer) is large enough to hold h tiles between chunks for typical head dims (K=V=128: tile = 128×128×4 = 64 KB — tight but possible)

**Rough speedup estimate:** 15–25% for the h-kernel specifically, dependent on seqlen. Higher impact at longer sequences where chunk count NT is large.

**Difficulty:** triton-hard — persistent kernel scheduling in Triton requires careful use of `tl.program_id()` remapping and atomic-based work queues. Requires Triton 3.2+ for reliable TMEM persistence.

---

### OPP-5: FP8 q/k accumulation path

**What changes:**  
Add an opt-in FP8 forward pass for q, k tensors in `chunk_gla_fwd_kernel_o` and the intra-chunk kernel. Since `use_qk_l2norm_in_kernel=True` already normalizes q and k to unit norm, values are bounded in [-1, 1] — ideal dynamic range for E4M3 FP8.

**Why it helps Blackwell:**  
- Blackwell 5th-gen Tensor Cores: FP8 throughput is ~2× FP16 (6,704 vs ~3,352 TOPS estimated)
- Load bandwidth halved for q, k tiles → better L2 hit rate, less DRAM bandwidth pressure
- MXFP8 (microscaling FP8) with per-32-element scale factors is natively supported on Blackwell; the L2-normalized q, k tiles have uniform dynamic range that maps well to MXFP8's block structure

**Tradeoff:** accumulation must remain in FP32 or BF16 to preserve delta rule numerical stability. The intra-chunk lower-triangular solve involves division, which is sensitive to quantization.

**Rough speedup estimate:** 20–40% for the output kernel if compute-bound (FP8 doubles MMA throughput). Smaller impact (~10%) if the kernel is memory-bandwidth-bound, which it likely is for small head counts.

**Difficulty:** requires CUDA — Triton FP8 support on sm_120 was partial as of early 2025. Full MXFP8 requires PTX-level intrinsics or CuTe. This is an upstream research item.

---

### OPP-6: TMA descriptors for K/V chunk loads

**What changes:**  
Replace manual `tl.load()` pointer arithmetic for K and V tiles with `tl.make_block_ptr()` + async TMA loads. This applies to `chunk_gla_fwd_kernel_o` and the intra-chunk kernels.

**Why it helps Blackwell:**  
- TMA offloads the address calculation and bounds checking from CUDA cores to dedicated hardware
- On Blackwell, TMA multicast allows one TMA descriptor to load the same V tile to multiple SMs simultaneously — relevant when multiple attention heads share the same KV cache (grouped-query scenario)
- Frees ~5–10% of CUDA core cycles currently spent on index arithmetic
- Reduces warp divergence from boundary checks

**Rough speedup estimate:** 5–10% for memory-bound kernels. Not transformative alone but synergistic with larger tiles (OPP-2, OPP-3).

**Difficulty:** triton-medium — Triton's `tl.make_block_ptr()` is already available and works on sm_120. Requires rewriting load loops to use block pointer idiom; fla-core partially uses this pattern in newer ops.

---

### Summary Table

| # | Opportunity | Speedup (kernel) | Difficulty | Blackwell-specific? |
|---|-------------|-----------------|------------|---------------------|
| OPP-1 | IS_NVIDIA_BLACKWELL autotune | 10–20% | triton-easy | Yes |
| OPP-2 | BT: 64→128 | 15–30% (intra) | triton-easy/medium | No (helps all) |
| OPP-3 | BV: 64→128/256 in h+output kernels | 20–35% (h+output) | triton-medium | Partly |
| OPP-4 | Persistent h-kernel | 15–25% (h-kernel) | triton-hard | Yes (TMEM) |
| OPP-5 | FP8 q/k path | 20–40% (output) | requires CUDA | Yes (5th-gen TC) |
| OPP-6 | TMA descriptors | 5–10% | triton-medium | Partly |

---

## Section 4 — Quantitative Back-of-Envelope Estimate

### Kernel time breakdown (estimated, no profiling available)

Using the dispatch graph from §1.1, a rough time breakdown for a single chunk_kda forward call at typical dimensions (B=1, T=2048, H=16, K=128, V=128, BT=64, seqlen in chunks = 32):

| Sub-kernel | Estimated % of chunk_kda time | Primary bottleneck |
|------------|-------------------------------|--------------------|
| `chunk_gla_fwd_kernel_o` (output) | ~40% | Compute (matmul BK×BV) |
| `chunk_gated_delta_rule_fwd_kernel_h` (h-state) | ~35% | Memory BW (h readback) |
| `chunk_kda_fwd_kernel_intra_*` (intra-chunk) | ~15% | Compute (triangular) |
| `kda_gate_chunk_cumsum_*` (gate) | ~7% | Memory BW |
| Backward kernels (relative to fwd) | ~2× fwd | — |

*These proportions are estimates based on FLOP counts and memory access patterns, not measured profiles.*

### Combined improvement calculation

Applying all six optimizations simultaneously (assuming they are partially independent):

- OPP-1 (autotune): 1.15× across most kernels
- OPP-2 (BT=128): 1.20× on intra-chunk (15% of total) → 1.03× system
- OPP-3 (BV=128/256): 1.25× on h+output (75% of total) → 1.19× system
- OPP-4 (persistent h): 1.20× on h-kernel (35% of total) → 1.07× system
- OPP-5 (FP8): 1.30× on output (40% of total) → 1.12× system
- OPP-6 (TMA): 1.07× across memory-bound kernels → 1.03× system

**Combined (geometric, accounting for overlaps):**  
1.15 × 1.03 × 1.19 × 1.07 × 1.12 × 1.03 ≈ **2.0× chunk_kda kernel speedup** (optimistic ceiling)

A conservative estimate (only OPP-1 + OPP-2 + OPP-3, the "triton-reachable" set):  
1.15 × 1.03 × 1.19 ≈ **1.41× chunk_kda kernel speedup**

### Translation to step-time speedup

From the prior MFU diagnosis context:

- Observed MFU ≈ 0.21% implies step time is dominated by communication, not compute
- FSDP AllGather/ReduceScatter over PCIe NVMe (no NVLink on RTX 5090): dominates ~60–70% of step time
- MLP GEMMs (compute-bound portion): ~20–25% of step time  
- KDA kernels (all layers combined): estimated **5–10% of step time**

**Net step speedup from 2× KDA kernel improvement:**  
Δ_step = 0.075 × (1 - 1/2.0) = **+3.75% step time reduction** (optimistic)  
Δ_step = 0.050 × (1 - 1/1.41) = **+1.45% step time reduction** (conservative, triton-only)

**Conclusion:** Even a perfect 2× KDA kernel improvement yields only ~2–4% step time improvement on PCIe-bottlenecked RTX 5090.

---

## Section 5 — Why This Is Not Our Scope

### 5.1 Correct Upstream Home

All kernel-level work described in §3 belongs in the `fla-org/flash-linear-attention` repository, not in our `torchtitan/experiments/kimi_linear/` integration. Our integration calls fla-core as a library dependency and does not fork or modify kernel code.

**Suggested upstream GitHub issue title:**  
> `KDA chunk kernel: Blackwell sm_120 (GB202/GB100) optimization opportunities — tcgen05, DSMEM, larger tiles, FP8`

**Issue body skeleton:**
- Architecture detection gap: no `IS_NVIDIA_BLACKWELL` branch in autotune configs
- OPP-1 through OPP-6 from §3 above
- Reference: Triton sm_120 backend status for tcgen05 lowering

### 5.2 Evidence Required Before Confirming Each Suggestion

To validate the hypotheses in §3, the following profiling evidence would be needed (tools: `nsys profile` + `ncu`):

| Suggestion | Required ncu metric | Threshold indicating opportunity |
|------------|--------------------|---------------------------------|
| OPP-1 (autotune) | `sm__warps_active.avg.pct_of_peak_sustained_active` | < 75% → warp count suboptimal |
| OPP-2 (BT=128) | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `sm__sass_thread_inst_executed_op_dfma_pred_on.sum` | High load/compute ratio → memory-bound intra kernel |
| OPP-3 (BV wider) | `sm__sass_thread_inst_executed_op_hfma_pred_on.sum` vs theoretical peak | < 60% utilization → tile too narrow |
| OPP-4 (persistent) | `dram__bytes_write.sum` for h-kernel | > 200 MB/fwd-pass → state writeback dominant |
| OPP-5 (FP8) | `sm__sass_thread_inst_executed_op_hfma_pred_on.sum` vs FP8 peak | Compute-bound with FP16 → FP8 doubles throughput |
| OPP-6 (TMA) | `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | High request count → address arithmetic overhead |

We currently have **no ncu profiles** of the KDA kernels. Generating them requires `sudo` access and a running training job with `ncu --target-processes all`.

---

## Section 6 — Interaction with Prior Diagnosis

### 6.1 Reconciling with the PCIe Bottleneck Claim

The prior claim — *"TPS 上不去主要因为 GEMM 维度太小 + PCIe 带宽太低；KDA triton kernel 是次要因素"* — is supported and refined by this analysis:

**Bottleneck order (PCIe-connected RTX 5090, FSDP=8):**

```
1. PCIe communication (AllGather 16–256 MB, ReduceScatter 16–256 MB)
   → ~60–70% of step wall time
   → Fixes: NVLink switch, NVLink-connected server Blackwell, reduce FSDP degree

2. MLP GEMM compute (small dimension problem)  
   → ~20–25% of step wall time
   → Model dim too small for peak tensor core utilization; roofline says bandwidth-bound
   → Fixes: larger model, sequence packing, higher GBS

3. KDA kernel (chunk_kda + fused_recurrent_kda)
   → ~5–10% of step wall time (estimated)
   → Confirmed as secondary bottleneck
   → Fixes: §3 optimizations (2–4% net step improvement at best)
```

**Verification of the KDA secondary claim:**  
The FLOP count of `chunk_kda` with T=2048, H=16, K=V=128, BT=64:
- Intra-chunk: O(T · BT · K) ≈ 2048 × 64 × 128 × 16 heads × 2 = 0.54 GFLOPs
- h-state update: O(T · K · V) = 2048 × 128 × 128 × 16 × 2 = 1.07 GFLOPs  
- Output: O(T · K · V) ≈ 1.07 GFLOPs  
- Total forward: ~2.7 GFLOPs per layer

For a 12-layer model: ~32 GFLOPs per forward pass for KDA. At RTX 5090 BF16 throughput (roughly 800–900 TFLOPs theoretical, effective ~100–200 TFLOPs given small tiles), this would take ~0.16–0.32 ms per forward. Compare to PCIe AllGather of 256 MB per FSDP step at 32 GB/s PCIe: 8 ms per AllGather. KDA is indeed secondary by ~25×.

### 6.2 The NVLink Counterfactual

On a future **NVLink-connected Blackwell cluster** (e.g., B200 NVLink 720 GB/s):
- AllGather of 256 MB: 256 MB / (720/8 GB/s) ≈ 0.28 ms (vs 8 ms on PCIe)
- PCIe bottleneck disappears; communication overhead drops to ~5% of step time
- **In this scenario, the bottleneck order flips:** KDA kernel and MLP GEMM efficiency become dominant
- The §3 optimizations (tcgen05, persistent kernel, FP8) would provide meaningful step-time improvement (~15–30%) on NVLink hardware
- This makes filing the upstream fla-core issue valuable now, for when the hardware situation improves

### 6.3 Summary Assessment

| Scenario | KDA bottleneck rank | KDA optimization value |
|----------|--------------------|------------------------|
| Current: RTX 5090 FSDP=8 via PCIe | #3 (secondary) | Low (1–4% net step gain) |
| Future: B200 NVLink cluster, same model | #1 or #2 | High (10–25% net step gain) |
| Current: single-GPU inference (no comms) | #2 (behind MLP) | Medium (5–15% throughput gain) |

**Bottom line:** The prior diagnosis was correct. On this specific hardware configuration, KDA kernel optimization is a valid long-term investment (upstream issue worth filing) but will not meaningfully move the MFU needle today. The 0.21% MFU is overwhelmingly a communication architecture problem (PCIe vs NVLink), not a kernel efficiency problem.

---

*Analysis performed: 2026-05-03. Based on fla-core source code as fetched from github.com/fla-org/flash-linear-attention main branch. NVIDIA Blackwell architecture data from Wikipedia GB202 article and RTX 5090 marketing specifications. No GPU profiling data available; estimates are theoretical.*
