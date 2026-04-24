# Problem B — Kimi Linear PP cross-stage cache adapter benchmark

**Question being answered:** Does our cross-stage cache adapter
(implemented in `torchtitan/experiments/attn_res/pipeline_adapter.py`
and wired into Kimi via `kimi_linear/pipeline_adapter.py`) preserve
loss equivalence and reduce per-stage communication when AttnRes is
combined with pipeline parallelism on the Kimi Linear architecture?

This is the SYSTEMS benchmark — the scientific "does AttnRes help?"
question is answered by Problem A (FSDP A/B). Here we hold the model
constant and vary parallelism strategy to demonstrate the fork's
distinguishing contribution: the adapter that lets PP+AttnRes scale
to deep pipelines without bloating cross-stage transfer to O(L·d).

## Three-arm comparison

| Arm | Parallelism | AttnRes cache | Purpose |
|---|---|---|---|
| `naive_pp` | PP=4 V=2 lps=2 Interleaved1F1B | OFF (`TORCHTITAN_ATTNRES_CACHE` unset) | upper bound on PP communication — every stage ships the full block stack |
| `adapter_pp` | PP=4 V=2 lps=2 Interleaved1F1B | ON (`TORCHTITAN_ATTNRES_CACHE=1`) | the contribution — only delta blocks ship cross-stage, cached + accumulated locally |
| `fsdp` | FSDP shard=4, no PP | not applicable (no stages) | reference — what running on a single-node-fits-FSDP box looks like; loss target for the two PP arms |

`naive_pp` and `adapter_pp` MUST produce loss curves that match
`fsdp` within the bf16+NCCL nondeterminism band. Phase 3 already
validated this on 175M Llama L=16 N=8 over 200K steps (max |Δ|=0.06,
inside the naive-vs-naive band of ~0.13). Problem B repeats that
validation on Kimi Linear L=16 N=8 with KDA + MLA + MoE in the mix.

## Configuration (apples-to-apples across all three arms)

| Knob | Value | Notes |
|---|---|---|
| Model | `kimi_linear_436m_block_attn_res` | 436M is paper-native L=16 (Table 2), no shape-massaging needed; matches Problem A's AttnRes arm exactly so insights cross-cite cleanly |
| num_blocks | 8 | paper N=8 recipe; 16/8 = 2 layers per block — also matches PP=4 V=2 lps=2 = 8 virtual stages, every block boundary lines up with a stage boundary |
| LR (peak) | 2.20e-3 | paper 436M default (Table 2) |
| LR schedule | warmup 500 + cosine decay (decay_ratio=0.8) | torchtitan default |
| SEQ_LEN | 2048 | hardware-constrained |
| LOCAL_BS / rank | 1 | required for PP=4 V=2 lps=2 to fit; FSDP arm matches for fairness |
| GLOBAL_BS | 4 | LOCAL_BS × NGPU |
| NGPU | 4 | 4× RTX 5090 |
| Mixed precision | bf16 (param + reduce) | torchtitan default |
| AC | off | Phase 4c |
| torch.compile | OFF for benchmark fairness | compile interacts with PP scheduling differently than FSDP-only; turn OFF on all three arms so the comparison isolates parallelism |
| use_grouped_mm | True | uniform across arms (KimiMoE default since 309b462) |
| STEPS | 1000 | enough for steady-state tps + memory + loss-alignment readout; this is a system benchmark, not a paper reproduction |

The PP arms additionally pin:
- `--parallelism.pipeline_parallel_degree 4`
- `--parallelism.pipeline_parallel_schedule Interleaved1F1B`
- `--parallelism.pipeline_parallel_layers_per_stage 2` (V=2 virtual stages, lps=2 layers per stage → 8 virtual stages → matches num_blocks=8)
- `--parallelism.data_parallel_shard_degree 1`

The FSDP arm pins instead:
- `--parallelism.pipeline_parallel_degree 1`
- `--parallelism.data_parallel_shard_degree 4`

## Metrics to collect (post-run)

Per arm:
1. `train.log` — every step's loss, grad_norm, tps, memory
2. `peak_memory_per_rank` from log
3. `step_time_steady_state` median over steps 100..1000
4. `loss_at_{1,100,500,1000}` for the alignment table

Cross-arm derived metrics:
- `|loss_naive - loss_adapter|` per logged step → max across run; should be inside bf16 band (~0.06-0.13 typical)
- `tps(adapter) / tps(naive)` → should be ≥ 1 (adapter saves comm)
- `tps(fsdp) / tps(adapter)` → expected > 1 on PCIe (FSDP wins on single-node), but the gap must be small enough that adapter doesn't make PP unusable
- `peak_mem(adapter) - peak_mem(naive)` → adapter trades a small per-rank cache memory for big comm savings; expected delta ~200-500 MB (same order as Phase 3's 175M Llama)

## Runs (each ~ 30 minutes wall on 4× RTX 5090)

1. `launch_naive_pp.sh` — baseline parallelism reference (PP without adapter)
2. `launch_adapter_pp.sh` — the contribution
3. `launch_fsdp_reference.sh` — FSDP-as-loss-target

Run sequentially (single 4-GPU box). Total compute: ~1.5h for all three.

## What this benchmark proves (and what it doesn't)

**Proves:**
- Adapter and naive PP produce identical loss curves on Kimi Linear
  (architectural correctness independent of attention type)
- Adapter reduces cross-stage communication vs naive PP
- Adapter overhead vs FSDP is bounded (so PP-with-AttnRes is viable
  in scenarios where FSDP can't fit)

**Does NOT prove:**
- That AttnRes helps Kimi Linear convergence (that's Problem A)
- That the adapter scales to deeper pipelines / multi-node (would
  need a real cluster); this benchmark is single-node 4-GPU on PCIe
- That adapter beats FSDP on throughput (it doesn't — and shouldn't,
  since FSDP is the right tool for single-node-fits scenarios). The
  point is that adapter beats *naive PP* and unlocks scenarios where
  PP is the only option.
