# RFC Issue Draft: Block Attention Residuals for torchtitan

> Copy this into a GitHub issue on pytorch/torchtitan

---

**Title**: [RFC] Add Block Attention Residuals (AttnRes) support

**Labels**: `enhancement`, `RFC`

---

## Summary

I'd like to implement [Block Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, March 2026) in torchtitan. AttnRes replaces fixed residual connections with learned, input-dependent depth-wise attention, enabling each layer to selectively aggregate earlier representations via softmax attention over depth.

Block AttnRes partitions layers into N blocks (~8), applies standard residuals within blocks, and uses attention only across block boundaries. This achieves equivalent training efficiency to 1.25x more compute (Table 2, paper) with <4% training overhead under pipeline parallelism and <2% inference latency overhead.

## Motivation

- **Strong empirical results**: +7.5 GPQA-Diamond, +3.6 Math, +3.1 HumanEval on Kimi Linear 48B (Table 3)
- **Practical drop-in replacement**: Only adds one RMSNorm + one pseudo-query vector (d-dimensional) per layer — negligible parameter increase
- **PP-friendly by design**: Block structure enables cross-stage caching that makes pipeline communication constant after first virtual stage
- **Validated at scale**: Trained on Kimi Linear (48B total / 3B activated, 1.4T tokens)

## Proposed Design

### Model-level changes only — no PyTorch pipelining modifications needed

The key insight is that `PipelineStage` already supports tuple tensor outputs via P2P send/recv. By modifying the model's `forward()` to return `(hidden_state, blocks_tensor)`, block summaries flow naturally between PP stages.

### Changes:

1. **New file**: `torchtitan/models/common/attn_res.py`
   - `block_attn_res(blocks, partial_block, proj, norm)` — core attention over block representations
   - `AttnResConfig` dataclass

2. **Modified**: `torchtitan/models/common/decoder.py`
   - `Decoder.forward()` accepts optional `blocks` tensor as second positional arg
   - When AttnRes enabled: threads blocks between layers, returns `(output, blocks_tensor)` for intermediate PP stages

3. **Modified**: `torchtitan/models/llama3/model.py`
   - `Llama3TransformerBlock` gains AttnRes parameters: `attn_res_proj`, `mlp_res_proj` (pseudo-queries, **zero-initialized**), `attn_res_norm`, `mlp_res_norm` (RMSNorm)
   - New `forward_attn_res()` method following paper's pseudocode (Figure 2)

4. **Config**: New flags `--model.use_attn_res`, `--model.attn_res_num_blocks`

### PP Integration:

- Stage 0: creates `blocks = [tok_embedding]`, returns `(partial_block, blocks_tensor)`
- Middle stages: receives `(partial_block, blocks_tensor)` as positional args, processes layers, returns updated tuple
- Last stage: applies final attention + norm + output, returns single logits tensor

Initial implementation uses naive approach (pass all blocks per transition). Cross-stage caching optimization can follow as a second PR.

### Key invariants:
- Pseudo-query vectors **must** be zero-initialized (ensures initial behavior equals uniform average = standard residual)
- `num_blocks` should align with PP degree for clean stage-block boundary mapping
- When `use_attn_res=False`, behavior is identical to current code (backward compatible)

## Validation Plan

1. **Single-card correctness**: Loss curve alignment with paper's small-scale points (Table 2)
2. **PP correctness**: Numerical match between single-card and PP=8,VP=2 on small model
3. **Benchmark**: Step time overhead, memory overhead, communication trace under interleaved 1F1B
4. **Target**: Llama3 1.5B-2B dense, PP=8, VP=2, FSDP inner, N_blocks=8, ~20B tokens

## References

- Paper: https://arxiv.org/abs/2603.15031
- Official repo (pseudocode only): https://github.com/MoonshotAI/Attention-Residuals
- Kimi infra engineer's implementation notes: https://www.zhihu.com/question/2016993095078684011/answer/2017381145474508331

## Questions for Maintainers

1. Should this start as an experiment (`torchtitan/experiments/attn_res/`) or go directly into core model code?
2. Preference on config structure — new `AttnResConfig` dataclass or flat flags on existing model config?
3. Any concerns about the tuple-output approach for PP integration?

I'm happy to start with a smaller PR (single-card only, no PP) and follow up with PP integration.
