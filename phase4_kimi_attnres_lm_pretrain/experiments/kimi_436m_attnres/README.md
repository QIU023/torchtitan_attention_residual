# Problem A — Kimi Linear AttnRes vs Baseline (FSDP A/B)

**Question being answered:** does Block AttnRes (paper recipe N=8) improve
training loss when layered onto Kimi Linear at the 436M scaling-law shape?

**Why FSDP, not PP+adapter:** these two runs are the SCIENTIFIC AB,
where parallelism is a confound, not the subject. FSDP is the right
parallelism for 4× RTX 5090 (model fits, comm cost low,
no pipeline bubble). The cross-stage cache adapter is independently
demonstrated in a separate "Problem B" systems benchmark — see
`phase3_attnres_pp_integration/` for the 175M Llama 200K-step naive vs adapter alignment
that already validated it; a Kimi-Linear flavored repeat of that
benchmark belongs in `phase4_kimi_attnres_lm_pretrain/experiments/kimi_pp_adapter/` (TBD).

## Configuration (apples-to-apples, both runs)

| Knob | Value | Source |
|---|---|---|
| Model size | 436M (L=16, d=1168, d_ff=528) | paper Table 2 |
| Architecture | Kimi Linear (KDA:MLA = 3:1, MoE every layer except first dense) | paper §5 |
| Vocab | 163,840 | paper / HF config |
| LR (peak) | 2.20e-3 | paper Table 2 (436M row) |
| LR schedule | warmup 500 + cosine decay (decay_ratio=0.8, min_lr_factor=0.1) | torchtitan default |
| Optimizer | AdamW | torchtitan default |
| SEQ_LEN | 2048 | hardware-constrained (paper: 8192) |
| LOCAL_BS / rank | 3 | max that fits 32GB with grouped_mm + compile |
| NGPU | 4 (RTX 5090) | local box |
| GLOBAL_BS (effective) | 12 | hardware-constrained (paper: 384) |
| grad_accum | 1 | none |
| Mixed precision | bf16 (param + reduce) | torchtitan default |
| FSDP shard degree | 4 | full shard, no replicate |
| Activation checkpoint | OFF | Phase 4c parallelize_kimi_linear |
| torch.compile | ON | post-Phase 4d (default in launcher) |
| use_grouped_mm | True | post-Phase 4d (default in KimiMoE) |
| STEPS | 12,500 | matches the active baseline run |

**Only knob that differs between the two runs:** `--config`. Baseline =
`kimi_linear_436m_baseline`, AttnRes = `kimi_linear_436m_block_attn_res`
(num_blocks=8, 2 layers per block, paper recipe).

## Deviations from paper (called out for completeness)

- SEQ_LEN: 2048 vs paper 8192 (4× shorter context). Paper SEQ=8192
  OOMs on 4×5090 even at LOCAL_BS=1.
- GLOBAL_BS: 12 vs paper 384 (32× smaller). Reaching paper batch via
  grad_accum=32 would take ~6 weeks at our throughput.
- Total tokens: 12,500 × 12 × 2048 = 307M vs paper 87.9B (0.35% of
  paper budget). This is a model-correctness AB validation, NOT a
  paper performance reproduction.

The deviations are identical between the two runs, so they cancel
out in the AttnRes-vs-baseline comparison. The downstream paper
performance numbers themselves require H100/H200/B200 multi-node
runs at paper-strict SEQ=8192 batch=384.

## Runs

1. `launch_baseline.sh` — kicks off `kimi_linear_436m_baseline` for
   12,500 steps. Already running as of 2026-04-23 22:49 UTC; see
   `../../runs/kimi_436m_baseline_fsdp_overnight/`.
2. `launch_attn_res.sh` — kicks off `kimi_linear_436m_block_attn_res`
   for 12,500 steps. Run after the baseline finishes.

## Comparison artifacts (to be produced after both runs)

- Loss curves overlaid with MA-50 smoothing (use the existing
  `phase3_attnres_pp_integration/plot_naive_vs_adapter.py` adapted for these two flavors)
- Per-window mean loss table (steps 1, 500, 1K, 2K, ..., 12.5K)
- Memory + throughput delta (AttnRes adds ~4 GiB block storage,
  +2-5% throughput in the steady state on 4×5090)

## Next experiments (out of scope here)

- **Problem B**: Kimi Linear PP=4 V=2 + adapter benchmark (3-way:
  naive PP / adapter PP / FSDP-as-reference) — showcases the fork's
  cross-stage cache adapter contribution. Separate folder.
- **Larger scale**: 528M / 48B-A3B AttnRes runs require H-class
  hardware. Same launcher works; just bigger memory footprint.
