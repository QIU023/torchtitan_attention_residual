# PR-able fixes for AttnRes inference NaN

After exhaustive testing on RTX 5090 (SM 12.0 / Blackwell):

## What we tried + result

| Backend | Status |
| --- | --- |
| `flashinfer_mla` (default for MLA) | NaN at layer 16 with our SFT'd ckpt |
| `torch_native` | Shape mismatch — doesn't support MLA's (q_a, kv_compressed) layout |
| `triton` | OOM (`shared memory: 131072 required, 101376 available`) |
| `fa3` | `requires SM>=80 and SM<=90`, RTX 5090 is SM 12.0 |
| `flashinfer` (bare) | Same MLA path, same NaN |

**Conclusion**: on Blackwell GPUs, the only working MLA backend is
`flashinfer_mla`, and it produces NaN on our model. Same model loaded
in torchtitan eager mode produces clean `max=10.69` logits.

## Why this isn't an algorithm bug

Same SFT step-2344 ckpt:
- torchtitan eager forward → clean output, no NaN
- SGLang flashinfer_mla forward → NaN at layer 16 attn

The Block AttnRes residual stream's per-block magnitude growth
(`max=77` by chunk 12) is consistent across both paths. It's how the
algorithm IS — production Kimi models presumably have similar growth
but their (1) model is bigger, (2) ckpt is more converged, (3)
production attention path uses fp32 for sensitive ops.

## PR-able fixes

### A. SGLang upstream issue (file as bug, not PR)

**Title**: `flashinfer_mla NaN with high-magnitude bf16 inputs (Block AttnRes)`

**Body**:
* Minimal repro: load Kimi Linear AttnRes (1.4B-active) with our overlay, run greedy on long-context prompt
* Same model + weights via torch eager: works
* Via flashinfer_mla: NaN at the model's deepest MLA layer with input max≈77
* Affects RTX 5090 / Blackwell deployments where alternative MLA
  backends are unavailable
* Suggested fixes:
  - (a) Add `--mla-fp32-scoring` flag — do QK matmul + softmax in fp32, V multiply in bf16
  - (b) Detect input magnitude > threshold, fall back to eager SDPA
  - (c) Document brittleness for non-converged models

**Effort**: file the issue + minimal repro: 1 hour. Fix on SGLang side: their team's call.

### B. Self-contained fix in our overlay (PR-ready, no upstream coord)

**Title**: `[VLM/AttnRes] fp32 eager SDPA fallback for MLA layers under high-magnitude residuals`

**Approach**: in `KimiAttnResDecoderLayer._run_attn`, detect:
1. Layer is MLA (full attention, not KDA)
2. `attn_input.abs().max() > FALLBACK_THRESHOLD` (e.g. 32)

If both true, bypass `self.self_attn(...)` (which routes to flashinfer_mla)
and instead:
1. Run RMSNorm in fp32
2. Compute `q_a`, `kv_compressed`, `q_pe`, `k_pe` projections in fp32
3. Manual SDPA: `softmax(Q @ K.T / sqrt(d_k))` in fp32
4. `V @ attention_weights` in bf16
5. Cast back to bf16

**Code skeleton** (mirroring `DeepseekMLAForwardMixin.forward_absorb_core`):

```python
def _run_attn_fp32_fallback(self, attn_input, positions, forward_batch):
    # fp32 RMSNorm + projections
    x = attn_input.to(torch.float32)
    ln_w = self.input_layernorm.weight.float()
    eps = self.input_layernorm.variance_epsilon
    rms = (x * x).mean(-1, keepdim=True).add(eps).rsqrt()
    h = x * rms * ln_w  # fp32

    sa = self.self_attn
    # MLA Q path: compress + decompress
    q_a = sa.q_a_proj(h.to(sa.q_a_proj.weight.dtype))[0]
    q_a = sa.q_a_layernorm(q_a)
    q = sa.q_b_proj(q_a)[0]  # [T, num_heads * (qk_nope + qk_rope)]
    # MLA KV path
    latent = sa.kv_a_proj_with_mqa(h.to(...))[0]
    kv_a, k_pe = latent.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    kv_a = sa.kv_a_layernorm(kv_a)
    kv = sa.kv_b_proj(kv_a)[0]
    # Split K_nope, V
    ...
    # Manual fp32 SDPA
    scores = (Q.float() @ K.float().transpose(-2, -1)) / math.sqrt(d_k)
    weights = scores.softmax(-1)
    attn_out = (weights @ V.float()).to(torch.bfloat16)
    return sa.o_proj(attn_out)[0]
```

**Effort**: 4-8 hours implementation + correctness verification (compare
against eager torchtitan output token-by-token). Doesn't require
SGLang upstream changes.

**Risk**: per-step latency ~3-5× slower for affected layers. Acceptable
for research / VLM PPO use case; not for production serving.

### C. Algorithmic fix: post-aggregation RMSNorm in `block_attn_res` (research PR, requires retrain)

**Title**: `[paper] Block AttnRes: bounded residual stream via post-aggregation normalization`

Add a final RMSNorm to `block_attn_res` output so the aggregated
residual is bounded layer-to-layer (matching standard pre-norm
transformer behavior).

**Effort**: 1-2 hours code change + full SFT retrain (~2-4h) + qualitative
eval to compare loss curves and inference robustness.

**Value**: paper-worthy if it shows equal or better convergence with
better inference numerical robustness.

## Recommended order

1. **A first** (1h, low risk): file the SGLang issue with minimal
   repro. Lets the SGLang team weigh in on whether they'd accept fp32
   scoring as an upstream feature.

2. **B in parallel** (4-8h): implement the self-contained fp32 MLA
   fallback in our overlay. This is the immediate unblock for VLM
   PPO without waiting for upstream.

3. **C as research follow-up** (4-6h): if B shows the model has good
   inference quality once flashinfer_mla bf16 is bypassed, then C is
   the proper paper-track contribution.

## Why this matters

Without B (or A landing upstream), VLM post-training on our 447M Kimi
AttnRes ckpt is blocked at the inference side. SFT works, training-
time forward works, but SGLang-served inference produces NaN logits =
no usable rollouts for GRPO/PPO.
