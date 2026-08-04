# Structure audit: our kimi_k3 vs #4025 vs the K3 report

Partial. Records what has been checked against the report text, what matches,
and one confirmed discrepancy. **Not a clean bill of health** -- the unchecked
rows at the bottom are unchecked, not passing.

Ground truth is `official_k3/k3_tech_report.pdf`. Where the two
implementations agree with each other but not the report, the report wins.

## Confirmed discrepancy: the final layer is not global attention

Report sec 2.1:

> Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1
> mixing ratio. This pattern is repeated throughout the backbone. **An
> additional Gated MLA layer is placed at the end of the backbone, ensuring
> that the final layer always performs global attention.**

Cross-check on the released shape: 93 layers = 23 x 4 + 1, i.e. 23 blocks plus
that one extra MLA. The reading holds.

| | layer pattern (13 layers) | final layer |
|---|---|---|
| report | 3 blocks of (3 KDA + 1 MLA), then 1 more MLA -> `{4,8,12,13}` | **MLA** |
| #4025 `_debugmodel` | `full_attention_layers = {4, 8, 12}` | **KDA** |
| our twin flavor | `full_attn_layers = [4, 8, 12]` (copied from theirs) | **KDA** |
| our own `model_configs.py` | `force_final_full_attn=True` -> `[4,...,92,93]` | **MLA** |

So our production path is right and has been since it was written -- the
helper's docstring quotes that exact report sentence. The twin flavor is
wrong, and it is wrong because it was built to mirror #4025 and mirrored this
too.

Fix is one entry: `full_attn_layers=[4, 8, 12, 13]` on the twin, with the KDA
list derived from it. That makes the twin differ from #4025's debugmodel by
exactly one layer, which is worth saying out loud rather than hiding, and is
also worth asking them about.

**Consequence for the published matrices**: every twin-flavor result so far
(3-step, 10-step, and the 100-step run in flight) is on a model whose final
layer is KDA. They are valid as "our parallelism on #4025's debug model" and
NOT valid as "our parallelism on the report's architecture". The headline
parallelism numbers should be re-run on the corrected flavor.

## Checked and matching

| item | report | #4025 | ours |
|---|---|---|---|
| SiTU-GLU gate branch | `b1*tanh(x/b1)*sigmoid(x)` | same | same |
| SiTU-GLU up branch | `b2*tanh(x/b2)` | same | same |
| SiTU betas | b1=4, b2=25 (fig. 4) | `beta=4.0, linear_beta=25.0` | `activation_situ_beta=4.0`, `..._linear_beta=25.0` |
| KDA:MLA ratio | 3:1 | 3:1 | 3:1 |
| AttnRes pseudo-query | `q_l = w_l` in R^d, score `q^T RMSNorm(k)` -> scalar | `attention_res_proj = Linear(dim, 1)` + `_norm(dim)` | `attn_res_proj` + `attn_res_norm` |
| AttnRes weighting | softmax over preceding layers, eq. 9 | matches | matches |
| Block AttnRes | L split into N blocks of S, partial sums | `attn_res_block_size=12` | block size 12 |
| loss function | -- | `torchtitan.components.loss.CrossEntropyLoss` | same class |

On the loss specifically, since it was asked: **neither side uses a home-made
loss**. Both use core torchtitan's `CrossEntropyLoss`. The only difference is
that #4025 wraps it in `ChunkedLossWrapper` and we do not, because our model
does not implement the `_skip_lm_head` forward that wrapper requires. Chunked
CE is the same quantity computed in chunks, so this is a memory optimization
rather than a numerical difference -- but it is a real structural difference
and implementing `_skip_lm_head` would remove it.

## Open: residual streams per layer

#4025's block carries **two** residual attentions:

    attention_res_norm / attention_res_proj
    ffn_res_norm       / ffn_res_proj

Ours carries **three**:

    attn_res_norm       / attn_res_proj
    mlp_res_norm        / mlp_res_proj
    final_attn_res_norm / final_attn_res_proj

The report's eq. 8-9 defines the mechanism once per layer `l`, over the
outputs of layers `1..l-1` plus the token embedding. Whether "layer" there
means the transformer layer or each sublayer is not settled by the text I have
read, and the extra `final_*` pair on our side is not accounted for at all.

**Not resolved. Do not treat either implementation as validated on this
point.** Resolving it needs the report's figure 1 (module diagram, p.5) read
alongside eq. 8-9, which is the next thing to do.

## Not yet checked

Listed so their absence is visible:

* Gated MLA: where the gate is applied, and whether `q_lora`/`kv_lora` ranks
  and head-dim splits match the report's table.
* Stable LatentMoE: the latent projection structure (report eq. 11), shared vs
  routed expert widths, and the normalization the report says accompanies it.
* Router: aux-loss-free bias rule vs the report's quantile balancing (sec
  2.3.3, eqs. 13-14) -- we have both, which is used by default matters.
* MoonViT-V2: factorized vs single attention pass, 2D RoPE variant, sd2_tpool,
  PatchMergerMLPV2.
* Norm placement and RMSNorm epsilon.
* Initialization: our twin starts at 12.07 against `ln(163840) = 12.006`;
  #4025's debugmodel starts at **12.478**, i.e. 0.47 nats above uniform, which
  means its initial logits are worse than random. Their loss is plain CE with
  no auxiliary term added (`load_balance_coeff` drives the router bias, not the
  loss), so the number is a real initialization signal. Cause not isolated --
  most likely the output projection's init scale, but that is a guess until
  measured.
