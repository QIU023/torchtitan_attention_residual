# Block AttnRes inference NaN bug — root cause analysis

## TL;DR

* **NOT** a vision-injection bug (despite earlier framing). Affects
  text-only LM inference identically.
* **Root cause**: Block AttnRes residual stream `partial_block` grows
  unboundedly across 4-layer blocks because the AttnRes aggregation
  *replaces* the standard pre-norm residual stream — losing the
  implicit per-layer normalization that keeps magnitudes in check.
  By block 4 (layers 12-15), `partial_block` reaches max ≈ 77 in
  bf16; the layer 16 (1-indexed) MLA self-attention then produces
  NaN logits.
* The same magnitude growth happened during SFT training; the LM
  weights converged on it (loss 2.22 → 1.22). But SGLang's
  inference path uses different attention kernels (flashinfer_mla)
  than torchtitan's training path, and these kernels produce NaN
  on our specific magnitude distribution.

## Diagnostic trace (LM-only, prompt "Hi", greedy)

Per-layer `partial_block` stats from `ATTNRES_NAN_TRACE=1` instrumentation:

```
embed_tokens                       abs_mean=0.04  max=0.18   ✅
== block 0 (layers 0-3, 1-idx 1-4) ==
L0_after_attn                      abs_mean=0.08  max=1.91   ✅
L0_after_mlp                       abs_mean=0.14  max=2.19
L1_after_attn                      abs_mean=0.20  max=2.47
L1_after_mlp                       abs_mean=1.27  max=10.25
L2_after_attn                      abs_mean=1.37  max=11.38
L2_after_mlp                       abs_mean=1.87  max=17.25
L3_after_attn                      abs_mean=1.97  max=19.00
L3_after_mlp                       abs_mean=3.07  max=37.50  ← block 0 commits here
== block 1 (layers 4-7) ==
L0_after_attn                      abs_mean=0.07  max=0.71
L3_after_mlp                       abs_mean=2.13  max=47.00
== block 2 (layers 8-11) ==
L0_after_attn                      abs_mean=0.10  max=0.84
L3_after_mlp                       abs_mean=2.63  max=77.50  ← growth visible
== block 3 (layers 12-15) ==
L0_after_attn                      abs_mean=0.31  max=4.59
L0_after_mlp                       abs_mean=1.52  max=48.75
L1_after_attn                      abs_mean=1.52  max=49.00
L1_after_mlp                       abs_mean=1.96  max=66.50
L2_after_attn                      abs_mean=1.99  max=68.00
L2_after_mlp                       abs_mean=2.43  max=76.50
L3_after_attn                      NaN            NaN        ← FAIL
```

**Layer 16 (1-idx, 0-idx 15) is MLA**. Layers (0-idx) 3, 7, 11, 15 are
all MLA per `full_attn_layers=[4,8,12,16]` config. Layers 11 and 15 are
both MLA-after-3-KDA-in-block. Layer 11's attn handled max=50, layer 15
fails on max=76. So it's a magnitude ceiling, not an architectural one.

## Symptoms it produces

* Greedy decode → `argmax(NaN_logits)` returns first index = token 0
  in Llama-3.1 vocab = `'!'`. Hence the all-`!!!!!` outputs.
* `return_logprob=True` shows `nan` at every position, all top-K
  alternatives `nan`.
* Intermittent — short prompts with mild residual growth might
  succeed (the magnitude trajectory varies with input distribution).
* Vision injection accelerates the problem (87× larger projector
  embeddings as input), but text-only LM also triggers given long
  enough prompts.

## What didn't fix it

| Attempt | Result |
| --- | --- |
| `disable_cuda_graph=True` | NaN unchanged (it's not a capture/replay issue) |
| `dtype=fp32` engine | sgl_kernel triton kernels mismatched-type errors |
| `dtype=fp16` | sgl_kernel mismatched-type errors |
| `attention_backend=triton` | flashinfer's MLA fallback triton kernel runs out of shared memory (131072 required, 101376 available on RTX 5090) |
| Project output 0.0115× scale-down (match text magnitude) | NaN unchanged |
| `partial_block` accumulated in fp32 | NaN unchanged (cast-back to bf16 happens before attn) |
| Manual fp32 RMSNorm at `input_layernorm` | NaN unchanged (issue is INSIDE self-attn, not at norm) |
| `partial_block.clamp(-20, 20)` before attn | NaN unchanged (clamp on partial-block doesn't affect the RMSNormed h that goes into attn) |

## Real fixes (require code beyond tonight's scope)

1. **Algorithmic**: change `block_attn_res` to return weighted sum of
   *normed* V instead of unnormed V — keeps magnitudes bounded layer-
   to-layer. Requires retraining the model with the new normalization.

2. **Per-layer rescale**: insert a learnable scaling factor on
   `attn_out` and `ffn_out` before adding to `partial_block`, so
   accumulation stays bounded. Same retraining requirement.

3. **fp32 attention path**: implement an fp32 MLA forward that bypasses
   flashinfer_mla. Significant SGLang patching.

4. **Pre-train Block AttnRes with depth-scaled residual** (e.g.
   `partial_block += attn_out / sqrt(num_layers)` like ReZero or
   StableT5). Requires retraining from scratch.

## What we have so far (committed)

* `ATTNRES_NAN_TRACE=1` instrumentation in
  `sglang/srt/models/attn_res_overlay.py` — per-block + per-layer
  magnitude stats logged via `_logger.warning`.
* `ATTNRES_BF16_ACCUM` / `ATTNRES_CLIP` / `ATTNRES_FP32_NORM` env
  toggles (none currently fix the bug, but useful for future probing).

## Path forward

For VLM post-training to actually work, we have to either retrain
with one of the algorithmic fixes (#1 or #2) or build out the fp32
MLA inference path (#3). All are multi-day engineering efforts.

The Block AttnRes overlay is *correct* in algorithm (loss converged
2.22 → 1.22 in SFT) but *not numerically robust* at inference depth.
The Kimi paper's original AttnRes formulation may have additional
normalization steps we missed, or paper-scale models may not hit the
overflow because they're trained with different optimizer states /
LR schedules that keep `attn_out` magnitudes smaller.

## CRITICAL UPDATE — SFT ckpt is FINE

Loaded the same SFT step-2344 ckpt via torchtitan eager-mode
`KimiLinearAttnResModel.forward(input_ids)` (the exact training-time
forward path). Output: **max=10.69, abs_mean=2.42, NO NaN**.

Same prompt, same weights, same algorithm — but SGLang inference
hits NaN at layer 16, while training-time forward produces moderate
logits (max=10.69 is normal for an unembedding output).

**Conclusion**: the SFT ckpt is fully functional. The bug is purely
in SGLang's inference path. Specifically the difference between:
- Training: torchtitan eager + `torch.nn.functional.scaled_dot_product_attention`
- Inference: SGLang + `flashinfer_mla` kernels

The `flashinfer_mla` kernel is producing NaN where eager SDPA
doesn't, on this specific magnitude distribution. This is a kernel-
side numerical issue, not an algorithmic bug.

## Why the original Kimi blog/paper has no NaN

Three convergent reasons (none requires retraining):

1. **Production Kimi K-series ckpt is much more converged** — Kimi K1.5/K2 
   are 100B+ params trained on T-scale tokens. Weights are settled;
   activation magnitudes likely don't grow to max=77.

2. **Production deployment likely uses fp32 for sensitive ops** — Kimi
   serving stack may force fp32 in RMSNorm divisor / attention softmax
   accumulator. SGLang's flashinfer_mla is bf16-only.

3. **Their head_dim is larger** — bigger attention head_dim means
   larger sqrt(d) scaling, producing smaller dot products that don't
   overflow as easily.

For us (small 1.4B-total model, 12,500 base steps + 2,344 SFT steps —
heavily undertrained vs production scale), magnitudes don't have the
same "convergence-bounded" property.

**Real fixes (no retraining needed)**:
A. Patch SGLang to do fp32 attention scoring — bypass flashinfer_mla
   for this specific model.
B. Use torch eager attention via SGLang's `attention_backend=torch_native`
   (not yet tried — separate from `triton` which OOM'd).
