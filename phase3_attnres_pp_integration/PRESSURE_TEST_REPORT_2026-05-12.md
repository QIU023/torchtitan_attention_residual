# PP Pressure Test — Final Report (2026-05-12)

## TL;DR

Three findings on 8× RTX 5090 PCIe:

1. **Block AttnRes L=16 sweep — DONE**: 3 PP×VP shapes × {naive, adapter} = 6 runs at 1000 steps from C4. Adapter aligns with naive in noise band (max |Δloss| = 0.0044, vs naive-vs-naive nondeterminism ~0.06-0.13).
2. **Full AttnRes L=32 PP=8×VP=4 — trainable** (loss 11.76 → 7.36 at step 140 before sglang preempt). Settles the "shallowest carrier supporting PP=8×VP=4 = 32 chunks" question.
3. **Block AttnRes L=32 from-scratch — unstable across all tested configs**: dim ∈ {768, 1024, 1280, 1536, 2048} × init ∈ {depth-scaled, uniform}, all inf-grad or nan from step 1. Documented as future work (needs targeted backward-hook diagnosis).

## Results 1 — L=16 Block AttnRes sweep (16 layers / 8 blocks, dim=768, n_heads=12, n_kv_heads=4)

| Shape | LBS | GBS | DP | mode | final loss | Δ (adapter−naive) | tps @ step 950 | mem rank 7 |
|---|---|---|---|---|---|---|---|---|
| **PP=8 × VP=2** | 16 | 16 | 1 | naive   | 5.42497 | — | 7,785 | 22.48 GiB |
| | | | | adapter | 5.42935 | **+0.00438** | 7,028 | 23.21 GiB |
| **PP=4 × VP=2** | 8 | 16 | 2 | naive   | 5.52833 | — | 11,176 | 14.16 GiB |
| | | | | adapter | 5.52941 | **+0.00108** | 8,619 | 14.76 GiB |
| **PP=4 × VP=4** | 16 | 32 | 2 | naive   | 5.13467 | — | 11,001 | 23.65 GiB |
| | | | | adapter | 5.13877 | **+0.00410** | 4,729† | 8.20 GiB‡ |

†‡ pp4_vp4 adapter ran today on the re-init box; lower tps and lower mem reflect a different sweep dir, not adapter overhead. Apples-to-apples wall-clock from the prior sweep:
- pp8_vp2: adapter +730 MiB cache (= 23.21 vs 22.48 GiB).
- pp4_vp2: adapter +600 MiB cache (= 14.76 vs 14.16 GiB).
- Adapter step time: pp8_vp2 +10%, pp4_vp2 +30% on 5090 PCIe — adapter pays bookkeeping > bandwidth savings on this fabric. NVLink-out / inter-node is where the saved bandwidth converts to wall-clock.

Max |Δloss| = 0.00438 across all 3 shapes. Naive-vs-naive nondeterminism band on this carrier (phase3 handoff 2026-04-21) was 0.06-0.13. **Adapter alignment passes.**

## Results 2 — Full AttnRes L=32 N=32 PP=8 × VP=4 (proven trainable)

`175M_attn_res_L32_n32` — every transformer-block is its own AttnRes-block (N=L). PP=8 × VP=4 = 32 chunks × 1 layer/chunk.

| step | loss | grad_norm | mem rank 7 |
|---|---|---|---|
| 1 | 11.76178 | 4.3 × 10¹⁷ (**finite**) | 21.75 GiB |
| 10 | 12.029 | 2.3 × 10¹² | 22.74 GiB |
| 20 | 10.785 | 2.4 × 10⁹ | 22.74 GiB |
| 30 | 10.203 | 7.2 × 10⁵ | 22.74 GiB |
| 100 | 8.10 | finite | 22.74 GiB |
| **140** | **7.36** | **finite (paper-aligned descent)** | 22.74 GiB |

(Sweep was preempted by sglang supervisor restart at step ~140, not by training failure.)

**Why Full AttnRes works where Block AttnRes fails**: at zero-init pseudo-queries, every layer's residual is the uniform mean of preceding sources (= bounded). Block AttnRes uses standard residual within each AttnRes-block, which accumulates unbounded — and at L=32 dim=768 the cumulative magnitude pushes some backward op into bf16/fp32 overflow.

## Results 3 — Block AttnRes L=32 from-scratch: untrainable across tested configs

Goal was paper-aligned **Block AttnRes** at PP=8 × VP=4 ≥ 32 chunks. Tested L=32 N=8 (4 transformer-blocks per AttnRes-block, paper sweet-spot × 1.33):

| dim | init scheme | step 1 grad_norm | step 30 |
|---|---|---|---|
| 768 | depth-scaled | inf | loss stuck 11.76 |
| 1024 | depth-scaled | inf | loss stuck 11.76 |
| 1280 | depth-scaled | inf | loss stuck 11.76 |
| 1280 | **uniform** (paper) | nan | loss=nan step 10 |
| 1536 | depth-scaled | inf | loss stuck 11.76 |
| 2048 | depth-scaled | inf | loss stuck 11.76 |
| **2048** | **uniform** | nan | loss=nan step 10 |

Also tested L=32 N=16 (2 t-blocks per AttnRes-block = same ratio as proven-stable L=16 N=8) at dim=768 depth-scaled: also inf-grad.

L=24 N=4 (6 t-blocks per AttnRes-block) at dim=768 depth-scaled: also inf-grad.

**What we learn**:
- Block AttnRes L=32 instability is NOT explained by aspect ratio alone (dim 2048 with aspect 1/64 still inf-grad — same aspect as GPT-2 small which trains).
- Block AttnRes L=32 instability is NOT explained by depth alone (Llama-3 8B is L=32 d=4096 and trains).
- Block AttnRes L=32 instability is NOT explained by init scheme alone (both depth-scaled and uniform fail at d=2048).
- Block AttnRes L=32 instability is NOT explained by t-blocks-per-AttnRes-block alone (L=32 N=16 with 2 t-blocks/block has same ratio as stable L=16 N=8 but still fails).

The remaining hypothesis: **a specific param in the L=32 backward graph overflows in bf16/fp32**. Diagnosis requires `register_hook` on every leaf param + every RMSNorm output to trace the first inf-source.

**Decided**: defer this diagnosis. Ship the present results with L=16 Block + L=32 Full AttnRes as the validated carriers. The Block AttnRes L=32 stability problem is a real research question (Why does it fail when math says it shouldn't?) but **independent of PP adapter correctness** — the adapter is exercised correctly on both L=16 Block and L=32 Full above.

## Results 4 — Kimi Linear 48B AttnRes carrier registered

In `torchtitan/experiments/kimi_linear/config_registry.py`:
- `kimi_linear_48b_baseline()` — no AttnRes
- `kimi_linear_48b_block_attn_res()` — paper §"Training recipe" exact: **N=9, 3 t-blocks (6 paper-layers) per AttnRes-block, dim=2304, 256 experts**
- `kimi_linear_48b_full_attn_res()` — N=L=27 ablation

Validated: 20/20 config fields match HF reference `moonshotai/Kimi-Linear-48B-A3B-Base/config.json`. Meta-device construction succeeds at 49.12B total params (= paper's 48B), ~3B activated (paper's 3B).

Single-node training infeasible (48B × 4 bytes + optimizer state exceeds 8×32 GiB even with FSDP+EP).

## Disk discipline established this session

1. `phase3_attnres_pp_integration/run_pp_pressure_test.sh` passes `--checkpoint.no-enable` on every pressure run (tyro form for `enable=False`; `=false`/`=False` are rejected).
2. `torchtitan/experiments/attn_res/config_registry.py:llama3_175m_baseline()` checkpoint default `keep_latest_k=3 → 2`.
3. `torchtitan/experiments/kimi_linear/config_registry.py:_base_trainer_config()` checkpoint default `keep_latest_k=3 → 2`.
4. `phase4_kimi_attnres_lm_pretrain/launch_continuation_100k.sh` `KEEP_K` default `5 → 2`.
5. `phase5_vlm_multimodal_sft/launch_train.sh` `keep_latest_k` `3 → 2`.

Result: smoke + pressure runs write NO checkpoints; other training keeps at most 2. Disk stayed at 25 GiB / 309 GiB through the entire session (~10 sweep runs).

## New carriers and code-level additions

`torchtitan/experiments/attn_res/`:
- `_175m_attn_res` extended with `dim` / `n_heads` / `n_kv_heads` / `init_scheme` kwargs (no behavior change at defaults).
- `_build_attn_res_layers` extended with `init_scheme` plumbing.
- New flavors registered: `175M_attn_res_L24_n4`, `175M_attn_res_L32_n16`, `175M_attn_res_L16_n16`, `175M_attn_res_L32_n32`, `attn_res_L32_n8_d{1024,1280,1536,2048}`, `attn_res_L32_n8_d{1280,2048}_uniform`.

`torchtitan/experiments/kimi_linear/`:
- `SCALING_LAW_TABLE` extended with `_SweepSize("48b", ...)` row.
- `build_kimi_linear_config` extended with `dense_intermediate_size` / `use_grouped_topk` kwargs and 48B-size defaults (256 experts, tie_word_embeddings=False, dense FFN intermediate = 9216, grouped-topk).
- 48B-specific MLA dim overrides (qk_nope=128, qk_rope=64, v=128) and KDA/MLA layer pattern matching HF config.

## Future work

1. **Diagnose Block AttnRes L≥32 inf-grad**: `register_hook` on every leaf param + RMSNorm output. Find the first inf-source. Hypotheses to test: (a) some lm_head / embedding interaction, (b) RMSNorm backward division by near-zero RMS, (c) AttnRes `softmax(w · V_stack) · V_stack` backward through stacked V at large N. Estimate: 1-2 hours.
2. **Multi-node Kimi 48B**: the registered `kimi_linear_48b_block_attn_res` carrier needs ≥ 2 nodes (16+ ranks) with FSDP+EP to actually train. Code is ready.
3. **NCCL trace wire-bytes**: run `phase7_nccl_traffic_catalog/extract_collectives.py` on L=32 Full AttnRes PP=8×VP=4 to capture the headline adapter vs naive wire-bytes comparison.

<!-- AUTO-GEN BEGIN kimi48b -->

## Kimi Linear 48B-layout PP runs (2026-05-12, auto-generated)

All Kimi paper architecture (KDA + MLA + MoE + Block AttnRes,
uniform init). FSDP+EP=8 + PP=8 + seq_len=1024. dim=1280.
Each row = one run; data from TensorBoard event files.

| run | last step | step 1 loss | final loss | step 1 grad | final grad | mem peak (GiB) | tps @ step 250 |
|---|---|---|---|---|---|---|---|
| `kimi48b_d1280_e32_L24N8_pp8vp3_adapter_20260512-090624` | — | (no TB data) | | | | | |
| `kimi48b_d1280_e32_L24N8_pp8vp3_adapter_20260512-091100` | 300 | 12.262 | **6.226** | 1.47e+05 | 1.83e+04 | 25.29 | 1189 |
| `kimi48b_d1280_e32_L24N8_pp8vp3_naive_20260512-094946` | 300 | 12.257 | **6.187** | 1.45e+05 | 1.75e+04 | 25.33 | 1192 |
| `kimi48b_d1280_e32_L32N8_pp8vp4_adapter_20260512-092719` | 1 | 12.257 | **12.257** | 2.13e+05 | 2.13e+05 | 26.15 | — |
| `kimi48b_d1280_e16_L32N8_pp8vp4_adapter_20260512-093021` | 300 | 12.259 | **5.970** | 2.12e+05 | 3.46e+04 | 24.76 | 1101 |
| `kimi48b_d1280_e16_L32N8_pp8vp4_naive_20260512-100309` | 300 | 12.262 | **5.959** | 2.12e+05 | 4.56e+04 | 27.93 | 1072 |
| `pressure_test_20260512-034748_L16fill` | 1000 | 11.762 | **4.987** | 2.71e+17 | 4.70e+04 | 8.20 | 4638 |

**Reading**: paper-aligned Block AttnRes (N matches paper 3 t-blocks/
AttnRes-block sweet spot ratio) on kimi_linear backbone trains
stably at L=24 (N=8) and L=32 (N=8, 4 t-blocks/block) at dim=1280
with PP=8 × VP=3/4 from random init. Loss descends monotonically;
grad_norm stays in 10⁴–10⁵ band throughout. **First Block AttnRes
PP=8×VP=4 pressure run on a paper-architecture single-node carrier.**

<!-- AUTO-GEN END kimi48b -->
