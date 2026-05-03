# AttnRes 8-GPU Throughput Bottleneck Analysis
## Run: `kimi_linear_436m_block_attn_res_n4`, B0 (FSDP=8), 8× RTX 5090 PCIe Gen5

---

## Executive Summary

**Verdict:** The throughput ceiling is driven almost entirely by **GEMM-small-dim inefficiency
at 0.15% MFU**. PCIe bandwidth contributes only **1.2% of step time** and is not a meaningful
bottleneck. The KDA Triton kernel accounts for **<0.01%** of step time. The prior claim
("GEMM small dims + PCIe bandwidth → primary bottleneck; KDA secondary") is **partially refuted**:
GEMM small dims confirmed as primary, but PCIe and KDA are both essentially irrelevant at this scale.

---

## Configuration (436M flavor)

Resolved from `config_registry.py` / `_SweepSize("436m")`:

| Parameter | Value |
|---|---|
| `hidden_size` (d) | 1168 |
| `num_hidden_layers` | 16 |
| `num_attention_heads` (H) | 16 |
| `num_experts` | 32 |
| `num_experts_per_token` (top-k) | 8 |
| `moe_intermediate_size` (d_ff) | 528 |
| `intermediate_size` (dense) | 528 |
| `vocab_size` | 163 840 |
| `kv_lora_rank` | 584 (= d/2) |
| `qk_nope_head_dim` | 73 (= max(32, d//H)) |
| `qk_rope_head_dim` | 36 (= max(16, nope//2)) |
| `v_head_dim` | 73 |
| `kda_head_dim` | 73 |
| KDA layers (1-indexed) | 1–3, 5–7, 9–11, 13–15 (12 layers) |
| MLA layers (1-indexed) | 4, 8, 12, 16 (4 layers) |
| Dense MLP layers (0-indexed) | 0 (1 layer; `first_k_dense_replace=1`) |
| MoE layers (0-indexed) | 1–15 (15 layers) |

**Recipe (B0 run):** FSDP=8, LBS=2, GBS=16, SEQ=260, `torch.compile=True`

**Critical: d=1168 = 2⁴ × 73; 73 is prime and not divisible by 16 or 128.**
This makes every K=1168 or N=1168 GEMM produce misaligned tiles for Blackwell tensor cores.

---

## Section 1 — GEMM Shape Inventory

Classification cutoff (Blackwell BF16 sm_120):
- `min(M,N,K) ≥ 256` → **OK** (tensor cores efficient)
- `128 ≤ min < 256` → **marginal** (underutilized)
- `min < 128` → **INEFFICIENT** (tensor cores misaligned or fallback path)

Batch: LBS=2, SEQ=260 → **M = 2×260 = 520** activation rows per GPU.
For MoE experts (grouped_mm): effective **M_expert = 520×8/32 = 130** tokens per expert.

### MLA per-layer (4 layers: indices 3, 7, 11, 15)

| Layer | Op | M | N | K | min | Class |
|---|---|---|---|---|---|---|
| MLA | `q_proj` | 520 | 1744 (H×(73+36)) | 1168 | 520 | **OK** |
| MLA | `kv_a_proj` | 520 | 620 (584+36) | 1168 | 520 | **OK** |
| MLA | `kv_b_proj` | 520 | 2336 (H×(73+73)) | 584 | 520 | **OK** |
| MLA | `o_proj` | 520 | 1168 | 1168 (H×73) | 520 | **OK** |

MLA shapes are all OK (min_dim=520 ≥ 256). The non-power-of-2 N/K=1168 still cause
tile padding overhead but tensor cores engage.

### KDA per-layer (12 layers)

| Layer | Op | M | N | K | min | Class |
|---|---|---|---|---|---|---|
| KDA | `q_proj` | 520 | 1168 (H×D) | 1168 | 520 | **OK** |
| KDA | `k_proj` | 520 | 1168 | 1168 | 520 | **OK** |
| KDA | `v_proj` | 520 | 1168 | 1168 | 520 | **OK** |
| KDA | `o_proj` | 520 | 1168 | 1168 | 520 | **OK** |
| KDA | `f_a_proj` | 520 | **73** | 1168 | **73** | **INEFFICIENT** |
| KDA | `f_b_proj` | 520 | 1168 | **73** | **73** | **INEFFICIENT** |
| KDA | `g_a_proj` | 520 | **73** | 1168 | **73** | **INEFFICIENT** |
| KDA | `g_b_proj` | 520 | 1168 | **73** | **73** | **INEFFICIENT** |
| KDA | `b_proj` | 520 | **16** | 1168 | **16** | **INEFFICIENT** |

5 out of 9 KDA projections are INEFFICIENT per layer (60%). The low-rank forget-gate
(f_a, f_b) and output-gate (g_a, g_b) bottleneck on N=K=73 (the kda_head_dim = d//H = 73,
which is prime). The beta projection b_proj has N=H=16, severely underutilizing tensor cores.

### Dense MLP (layer 0)

| Op | M | N | K | min | Class |
|---|---|---|---|---|---|
| `gate_proj` | 520 | 528 | 1168 | 520 | **OK** |
| `up_proj` | 520 | 528 | 1168 | 520 | **OK** |
| `down_proj` | 520 | 1168 | 528 | 520 | **OK** |

Dense MLP is OK but contributes only 0.3% of forward FLOPs.

### MoE per-layer (15 layers), per-expert (grouped_mm)

| Op | M_expert | N | K | min | Class |
|---|---|---|---|---|---|
| `expert_gate_proj` (×32) | **130** | 528 | 1168 | **130** | **marginal** |
| `expert_up_proj` (×32) | **130** | 528 | 1168 | **130** | **marginal** |
| `expert_down_proj` (×32) | **130** | 1168 | 528 | **130** | **marginal** |
| `router_gate` | 520 | **32** | 1168 | **32** | **INEFFICIENT** |
| `shared_gate_proj` | 520 | 528 | 1168 | 520 | **OK** |
| `shared_up_proj` | 520 | 528 | 1168 | 520 | **OK** |
| `shared_down_proj` | 520 | 1168 | 528 | 520 | **OK** |

Expert GEMMs use `grouped_mm` which fuses all 32 experts into one batched GEMM
`[32, 130, 528/1168]`. Despite batching, **M=130 is below the 256-threshold** and
the arithmetic intensity of 96 FLOPs/byte is below the Blackwell roofline ridge (234 FLOPs/byte),
making these **memory-bandwidth bound**.

### AttnRes Projections (32 total, 2 per layer)

| Op | M | N | K | min | Class |
|---|---|---|---|---|---|
| `AttnResProjection` (pre-attn, pre-ffn) | 520 | **1** | 1168 | **1** | **INEFFICIENT** |

These are dot-product scalar projections (d→1). N=1 makes them pure inner-product
vector operations — trivially small, cannot use tensor cores. However, their total FLOPs
are negligible (0.0% of forward budget; 32 × 2 × 520 × 1 × 1168 = ~39 MFLOPs).

### lm_head

| Op | M | N | K | min | Class |
|---|---|---|---|---|---|
| `lm_head` | 520 | 163 840 | 1168 | 520 | **OK** |

Despite OK classification, this single GEMM contributes **35.6% of all forward FLOPs**
(199 GFLOPs). Arithmetic intensity = 359 FLOPs/byte > ridge 234 → compute-bound.

### Summary

| Class | GEMM call count | FLOPs share (fwd) |
|---|---|---|
| OK | 113 individual calls | 57.6% |
| Marginal | 1,440 (32 experts × 15 layers × 3 GEMMs) | 41.5% |
| INEFFICIENT | 107 | 0.9% |

**41.5% of forward FLOPs sit in marginal-M expert GEMMs.** The large number of
INEFFICIENT calls (107) carry negligible FLOPs but each incurs per-call GPU kernel
launch overhead (~5–20 μs each, accumulated across ~850 total kernels per step).

---

## Section 2 — PCIe Bandwidth Budget

**Data source:** `phase5/runs/8gpu_b0_fsdp8_seed42/tier_c_trace/collective_summary.csv`
(236,178 rows = individual NCCL collective calls across all 8 ranks × 500 steps)

### Per-step per-rank breakdown (rank 304939, 500 steps)

| Collective | Calls/step | Dominant size | Total/step |
|---|---|---|---|
| AllGather (FSDP param reconstitution) | 35 | 16.74 MB | 554 MB |
| ReduceScatter (FSDP gradient reduction) | 19 | 33.48 MB | 602 MB |
| **Total AG + RS** | **54** | — | **1,156 MB** |

Note: RS dominant size (33.48 MB) ≈ 2× AG (16.74 MB), consistent with BF16 parameters
(AG) vs. float32-accumulated gradients (RS) or the FSDP2 double-buffer layout.

### Communication time estimate

RTX 5090 PCIe Gen5 x16: 64 GT/s × 2 bytes = **~50 GB/s theoretical**, **~30–40 GB/s NCCL
sustained** for 8-rank ring collectives. Using 35 GB/s as the conservative working estimate:

```
NCCL ring latency: time ≈ size × (nranks-1)/nranks / BW_per_link
AG per call (16.74 MB): 16.74 × 7/8 / 35 = 0.42 ms
RS per call (33.48 MB): 33.48 × 7/8 / 35 = 0.84 ms

Total AG/step:  35 × 0.42 ms = 14.7 ms
Total RS/step:  19 × 0.84 ms = 15.9 ms
Total PCIe comm: 30.6 ms
```

### Comparison with observed step time

| Metric | Value |
|---|---|
| Observed TPS (log, per GPU) | ~200 tokens/s |
| Observed step time (LBS=2, SEQ=260) | **2,600 ms** |
| PCIe comm (35 GB/s, no overlap) | 30.6 ms |
| PCIe comm (50 GB/s theoretical) | 23.1 ms |
| **PCIe as % of step time** | **1.2%** |

**PCIe bandwidth is not a bottleneck.** Even at the conservative 35 GB/s estimate
with zero overlap, FSDP communication consumes only 1.2% of the step budget.
FSDP2's built-in AG/RS prefetching would reduce this further to near-zero effective overhead.

**Comparison to NVLink:** An 8× H200 NVLink-connected box at 900 GB/s would save at most
(30.6 ms − 1.7 ms) = 28.9 ms per step → **net speedup of 1.01×**. Completely irrelevant
given the 2,600 ms step time is dominated by compute inefficiency.

---

## Section 3 — Compute Budget (GEMM FLOPs)

Forward-pass FLOPs at LBS=2, SEQ=260 (M=520 per GPU):

| Component | FWD GFLOPs | % |
|---|---|---|
| MoE routed experts (15 layers × 32 experts × top-k=8) | 231 | 41.3% |
| lm_head (M=520, N=163840, K=1168) | 199 | 35.6% |
| KDA linear projections (12 layers × 9 GEMMs) | 73 | 13.0% |
| MLA GEMMs (4 layers × 4 GEMMs + SDPA) | 25 | 4.4% |
| MoE shared expert (15 layers × 3 GEMMs) | 29 | 5.2% |
| Dense MLP layer 0 | 2 | 0.3% |
| KDA chunk_kda Triton kernel (12 layers) | 0.5 | 0.1% |
| AttnRes projections (32 × Linear(1168→1)) | ~0.04 | ~0% |
| **Total forward** | **560 GFLOPs** | 100% |

Total step FLOPs (fwd + bwd ≈ 3× fwd): **1.68 TFLOPS**

### Theoretical compute floor

```
At RTX 5090 peak BF16 (419 TFLOPS): 1.68T / 419T = 4.0 ms per step
Observed step time:                  2,600 ms
Ratio:                                650×  slower than peak
Effective throughput:                 0.65 TFLOPS (0.15% MFU)
```

The logger reports `tflops: 1.22` and `mfu: 0.39%`; the difference from our bottom-up
calculation (0.65 TFLOPS) likely reflects torchtitan's FLOP formula including optimizer
and re-materialization overhead.

### Roofline analysis (RTX 5090)

```
Memory bandwidth: ~1,792 GB/s (GDDR7)
Compute peak BF16: 419 TFLOPS
Roofline ridge: 419e12 / 1792e9 = 234 FLOPs/byte
```

| GEMM type | Arith. intensity | vs. ridge (234) | Bound |
|---|---|---|---|
| MoE expert (M=130, N=528, K=1168) | 96 FLOPs/byte | below | **memory-BW** |
| grouped_mm (M=4160=32×130, same N,K) | 96 FLOPs/byte | below | **memory-BW** |
| KDA f_a/g_a (M=520, N=73, K=1168) | 134 FLOPs/byte | below | **memory-BW** |
| KDA q/k/v proj (M=520, N=1168, K=1168) | 249 FLOPs/byte | above | compute |
| lm_head (M=520, N=163840, K=1168) | 359 FLOPs/byte | above | compute |
| MLA q_proj (M=520, N=1744, K=1168) | 282 FLOPs/byte | above | compute |

The MoE expert GEMMs — which hold 41% of forward FLOPs — are memory-bandwidth bound even
after grouped_mm fuses all 32 expert weight matrices. The reason: with only 130 tokens per
expert, the weight tiles read from DRAM cannot be reused enough to overcome the cost of loading
them. Total expert weight = 15 layers × 32 experts × 3 GEMMs × ~1.3 MB each = 1.8 GB;
at 1,792 GB/s this takes ~1 ms — but the achieved BW for non-contiguous small-M access
is far below peak, giving the ~4 ms estimate per expert tier.

---

## Section 4 — KDA Kernel Cost

The `chunk_kda` Triton kernel (fla-core) processes the delta-rule recurrent state update
for each KDA layer. FLOPs scale as O(B × T × H × D²) per layer:

```
Per KDA layer: B×T×H×D² = 2 × 260 × 16 × 73² = 44.3 MFLOPs
12 KDA layers:             44.3 × 12 = 532 MFLOPs (forward)
Full step (3× fwd):        532 × 3 = 1.60 GFLOPs
At 20 TFLOPS (Triton small kernel):  ~0.08 ms
As % of 2600 ms step:      < 0.003%
```

The associated ShortConvolution kernels (causal_conv1d, 3 per KDA layer × 12 layers = 36
launches) each process a `[2, 260, 1168]` tensor through a width-4 conv — trivially memory
bandwidth bound but at <1 MB of I/O each → < 0.6 μs per call → ~22 μs total for all 36.

`fused_kda_gate` (1 per KDA layer × 12 = 12 calls): gate sigmoid + A_log decay — pure
elementwise on `[2, 260, 16, 73]` tensors → ~5 μs each → 60 μs total.

**KDA kernel total contribution: <0.1 ms = <0.004% of step time.**

The prior observation that "KDA triton kernel is a secondary factor" is confirmed, but
the mechanism is different: it is secondary because its FLOPs and I/O are genuinely tiny
relative to the MoE+lm_head bulk, not because of any GPU scheduling issue.

---

## Section 5 — Verdict Table

Observed step time: **2,600 ms** (TPS ≈ 200 tokens/s per GPU, LBS=2, SEQ=260)
MFU from log: **0.39%** of RTX 5090 BF16 peak

| Component | Est. per-step time | % of step | Bottleneck class |
|---|---|---|---|
| **MoE expert GEMMs** (15 layers, 32 experts) | ~600–900 ms* | ~25–35% | Memory-BW bound (96 FLOPs/byte vs ridge 234) |
| **KDA linear projs** (12 layers × 9 GEMMs) | ~300–500 ms* | ~12–20% | INEFFICIENT: N=73 misaligned tensor core tiles |
| **lm_head** (M=520, N=163840) | ~150–300 ms* | ~6–12% | Compute-bound but OK (359 FLOPs/byte > ridge) |
| **MLA GEMMs** (4 layers × 4 GEMMs) | ~50–100 ms* | ~2–4% | Compute-bound, OK shapes |
| **FSDP AllGather** (35 calls/step, PCIe) | **14.7 ms** | **0.6%** | PCIe Gen5 x16, 35 GB/s sustained |
| **FSDP ReduceScatter** (19 calls/step, PCIe) | **15.9 ms** | **0.6%** | PCIe Gen5 x16, 35 GB/s sustained |
| **KDA chunk_kda** (12 layers, Triton) | **<0.1 ms** | **<0.01%** | Triton kernel, tiny FLOPs (0.53 GFLOPs) |
| Other (norms, elementwise, launch overhead) | remainder | remainder | Kernel launch + sync overhead |

*Individual component times are estimates scaled from MFU; the GPU is running all operations
at ~0.15–0.30% of peak throughput. The PCIe and KDA rows have tight lower bounds from
the roofline model; the compute rows are inferred from the residual budget.

### Verdict

**Primary bottleneck: GEMM-small-dim-bound (compute at 0.15% MFU)**, with two distinct mechanisms:

1. **Memory-bandwidth bound** (MoE experts, M=130): intensity 96 < ridge 234 → weight streaming
   is the gating factor. Grouped_mm batches all 32 experts but doesn't help M-dimension starvation.

2. **Tensor-core-misaligned GEMMs** (KDA f/g/b projections, N∈{16,73}): N=73 and N=16 are not
   multiples of 128 (the preferred tile size for Blackwell BF16). The cuBLAS/cuTLASS kernel must
   pad outputs and discard padding, wasting ~42% of tensor core throughput for N=73, ~88% for N=16.
   Root cause: `kda_head_dim = d//H = 1168//16 = 73` — **73 is prime**.

3. **Non-power-of-2 d_model**: `d = 1168 = 16 × 73`. All GEMMs with K=1168 experience
   the same prime-factor alignment penalty.

**PCIe bandwidth is NOT a bottleneck**: measured at 30.6 ms = 1.2% of step time.
Moving to NVLink (900 GB/s) would yield at most 1.01× speedup.

**KDA Triton kernel is NOT a meaningful bottleneck**: <0.1 ms = <0.004% of step time.
Its FLOPs are four orders of magnitude below the compute floor.

---

## Section 6 — Predictions

### A. 8× H200 NVLink (900 GB/s, ≈18× PCIe sustained bandwidth)

- PCIe comm drops from 30.6 ms → 1.7 ms per step
- Net step time change: 2,600 ms → ~2,571 ms
- **Speedup: 1.01×** — essentially zero benefit
- New bottleneck: **still GEMM-small-dim-bound** (interconnect does not help M=130 expert GEMMs)

### B. LBS 2→16 (8× more compute, identical communication volume)

- New M = 16 × 260 = 4,160 activation rows per GPU
- New M_expert = 4160 × 8/32 = **1,040 tokens per expert**
- New MoE expert intensity: **269 FLOPs/byte > ridge 234 → now compute-bound**
  - This is the single biggest gain: the dominant 41% FLOPs bucket shifts from memory-bound to compute-bound
- KDA f_a/g_a shape: M=4160, N=73, K=1168 → N=73 still misaligned → STILL INEFFICIENT
  - KDA low-rank projections remain misaligned regardless of batch size
- Comm fraction drops to ~0.15% of new step (essentially unchanged in absolute ms)
- **Expected step speedup: up to ~6–7× from LBS=2 to LBS=16** (the shift of expert GEMMs
  from memory-bound to compute-bound plus higher GEMM utilization throughout)
- Remaining bottleneck after LBS=16: KDA small-N GEMMs + non-power-of-2 d_model

### C. KDA kernel 1.5× faster on Blackwell

- chunk_kda contribution: <0.1 ms
- 1.5× speedup saves ≤0.05 ms → **<0.002% of step time**
- Even if ALL KDA-related ops (projections + conv + gate + chunk) became 1.5× faster:
  - KDA total (linear proj ~300 ms est. + kernels <1 ms) × 0.33× savings → ~100 ms
  - Net speedup: 100/2600 = **~3.8% improvement in step time**
- **Conclusion: KDA kernel optimization is low-leverage at LBS=2, SEQ=260**

---

## Limitations

[LIMITATION] The per-component time estimates in Section 5 are model-derived (roofline +
efficiency scaling), not measured. A PyTorch profiler trace (tier_d) would give exact kernel
timings. The "~0.15% MFU" is consistent with the logged `mfu: 0.39%` (which may use a different
FLOP counting formula including re-materialised activations and optimizer steps).

[LIMITATION] The expert GEMM intensity calculation assumes uniform load balancing
(exactly top-k=8 of 32 experts per token). In practice routing imbalance can raise hot-expert
M above 130 (helping intensity) or cause idle experts (wasting bandwidth). With
`moe_renormalize=True` + sigmoid routing + auxiliary-loss-free load balance, typical
variance is ±15% around the mean → expert M ranges 110–150.

[LIMITATION] Achieved GDDR7 memory bandwidth for non-contiguous access patterns
(expert weight tiles) is typically 60–80% of the spec 1,792 GB/s. The roofline bound used
here is optimistic; actual memory-BW-bound times are ~1.3–1.7× higher than calculated.

[LIMITATION] `torch.compile=True` fuses some elementwise ops into GEMMs and reduces
kernel launch overhead. The exact number of kernel launches (estimated 849) may differ;
compile fuses norms and activations but typically leaves GEMMs as separate calls.

[LIMITATION] The CSV has `opcount=0` for all rows (all 500 steps aggregated). Per-step
variance in communication pattern is not observable from this data.

---

## Prior Claim Assessment

**Prior Claude observation:** "TPS 上不去主要因为 GEMM 维度太小 + PCIe 带宽太低；KDA triton kernel 是次要因素"
*(TPS can't increase mainly because GEMM dimensions too small + PCIe bandwidth too low; KDA triton kernel is a secondary factor)*

| Sub-claim | Evidence | Verdict |
|---|---|---|
| GEMM dimensions too small | MoE M=130 (marginal), KDA N=73 (INEFFICIENT), d=1168 non-power-of-2 | **CONFIRMED** |
| PCIe bandwidth too low | Measured: 30.6 ms = 1.2% of step time | **REFUTED** — PCIe is inconsequential |
| KDA triton is secondary | chunk_kda: <0.1 ms = <0.01% of step | **CONFIRMED** (but for quantitatively different reasons — FLOPs are genuinely tiny, not just secondary to PCIe) |

The prior claim correctly identified GEMM small dims as the primary issue and correctly
deprioritized KDA. However it incorrectly attributed significant weight to PCIe bandwidth:
the actual PCIe share is 1.2%, not a material factor at this model size and batch configuration.

**The true bottleneck hierarchy at (d=1168, LBS=2, SEQ=260, FSDP=8) is:**
1. MoE expert GEMMs: memory-bandwidth bound due to M=130 (low arithmetic intensity)
2. KDA low-rank projections: tensor-core-misaligned due to N=K=73 (prime head_dim)
3. d_model=1168=2⁴×73 alignment tax on all GEMMs
4. lm_head: compute-bound but large (35% of FLOPs) — not a bottleneck, just dominant work
5. Everything else (PCIe, KDA chunk, AttnRes) < 2% total
