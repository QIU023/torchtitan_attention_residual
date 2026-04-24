# Problem B — Kimi Linear PP cross-stage cache adapter (single arm)

**Question being answered:** Does our cross-stage cache adapter
(implemented in `torchtitan/experiments/attn_res/pipeline_adapter.py`
and wired into Kimi via `kimi_linear/pipeline_adapter.py`) preserve
loss equivalence with the FSDP reference when AttnRes is combined
with pipeline parallelism on the Kimi Linear architecture?

This is the SYSTEMS run. The scientific "does AttnRes help?"
question is answered by Problem A (FSDP A/B in
`../kimi_436m_attnres/`).

## Scope (simplified from the original three-arm plan)

We only run **one PP arm** here — `adapter_pp` at full 12,500-step
length, matched to Problem A's two FSDP runs (436M baseline + 436M
block_attn_res N=4). The two arms originally planned and dropped:

* `naive_pp` — baseline parallelism reference. Phase 3 already
  validated naive-PP-vs-adapter-PP loss equivalence on 175M Llama
  L=16 N=8 over 200K steps. We don't need to repeat the proof on
  Kimi; the adapter-vs-naive comm comparison is documented elsewhere.
* `fsdp_reference` — duplicates Problem A's AttnRes FSDP run
  (same model, same hyperparameters, just same parallelism). No
  reason to re-run; we use Problem A's output as the loss target.

## Configuration

Identical to Problem A's AttnRes arm except for the parallelism
strategy:

| Knob | Value | Notes |
|---|---|---|
| Model | `kimi_linear_436m_block_attn_res` | L=16, num_blocks=8 (paper N=8 recipe; LOCAL_BS=1 means we have memory headroom for N=8 cache, no need for the n4 flavor here) |
| LR (peak) | 2.02e-3 (paper template via 528m base) | inherited from `_base_trainer_config` template; lr scheduling matches Problem A's 436M template the trainer config uses |
| SEQ_LEN | 2048 | hardware-constrained (paper: 8192) |
| LOCAL_BS / rank | 1 | required for PP=4 V=2 lps=2 to fit; FSDP arm matches |
| GLOBAL_BS (effective batch) | 12 | matches Problem A's effective batch (apples-to-apples) AND `num_microbatches = 12 / 1 = 12 ≥ 8 virtual stages` so Interleaved1F1B fills the pipeline + 4 microbatches of slack |
| NGPU | 4 (RTX 5090) | local box |
| pipeline_parallel_degree | 4 | |
| pipeline_parallel_schedule | Interleaved1F1B | adapter prerequisite |
| pipeline_parallel_layers_per_stage | 2 | V=2 virtual stages × lps=2 = 8 virtual stages, every block boundary lines up with a stage boundary |
| TORCHTITAN_ATTNRES_CACHE | 1 | adapter ON — only delta blocks ship across stages |
| Mixed precision | bf16 (param + reduce) | torchtitan default |
| AC | off | Phase 4c |
| torch.compile | OFF | compile + PP scheduling interaction noisy; off keeps the adapter measurement clean |
| use_grouped_mm | True | uniform with Problem A |
| STEPS | 12,500 | matches Problem A length |

## Comparison artifacts (post-run)

* Loss curves: Problem A's two FSDP arms (baseline +
  block_attn_res_n4) overlaid with Problem B's adapter_pp run.
* Throughput: tps for Problem A AttnRes FSDP vs Problem B
  adapter_pp.
* Memory: peak rank memory comparison.
* Adapter cache size at steady state (per-rank cache copies × bytes).

## Auto-chain

`run_after_baseline.sh` polls Problem A's AttnRes FSDP train.log
for "Training completed" and then launches `launch_adapter_pp.sh`
when the GPUs free up. Designed so a single overnight launch covers
both Problem A's AttnRes arm and Problem B sequentially.
