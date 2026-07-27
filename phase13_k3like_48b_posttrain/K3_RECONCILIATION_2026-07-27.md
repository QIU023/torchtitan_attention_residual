# K3 official release: full reconciliation (2026-07-27)

Executes the pre-registered checklist in
[K3_RELEASE_IMPACT_2026-07-16.md](../K3_RELEASE_IMPACT_2026-07-16.md) sec 4.
Every number below comes from the stored artifact
[official_k3/config.json](official_k3/config.json) or from the modeling code
quoted in [official_k3/SOURCE.md](official_k3/SOURCE.md) -- nothing here is
recalled or inferred.

## 0. Scorecard: what our implementation got right, and what it missed

| pre-registered question (sec 4) | official answer | our port |
|---|---|---|
| AttnRes block count N | `attn_res_block_size: 12`, commit at `layer_idx % 12 == 0` -> layers 0,12,...,84 = **8 blocks** over 93 layers | **our bet was "N ~= 8 sweet spot"** -- correct |
| Block vs Full AttnRes | **Block** (N=8 << L=93) | Block is our headline variant; both implemented |
| Pseudo-query design | `nn.Linear(hidden_size, 1, bias=False)`, **per layer** | `AttnResProjection` is exactly D->1 per layer -- **match** |
| Pseudo-query init | standard `_init_weights`, std=`initializer_range` | we **zero-init** on purpose (graft identity anchor). Deliberate divergence, see sec 4 |
| AttnRes placement | **twice per layer**: `self_attention_res_proj` pre-attn, `mlp_res_proj` pre-FFN | 2x per layer, second name identical -- **match** |
| Aggregation math | `scores=(k*w).sum(-1)`, `probs=scores.softmax(-1)`, `out=matmul(probs,v)` | softmax over blocks + weighted sum -- **match** |
| Gated MLA | `mla_use_output_gate: true`; `g = g_proj(h).sigmoid(); attn_out = attn_out * g` | we ship a provisional gate -- **verify form** (sec 3.2) |
| KDA gating | `use_full_rank_gate: true`, `gate_lower_bound: -5.0` | we use fla-core's default gate -- **gap** (sec 3.3) |
| Quantized checkpoint import | `mxfp4-pack-quantized`, group 32, `scale_dtype uint8`, **routed experts only** | our packed-MXFP4 path exists; the *scope* is wrong (sec 5) |
| Inference two-phase | not yet compared | pending |

## 1. Text model: official vs our provisional `2p8t` flavor

Our `2p8t` row was declared PROVISIONAL with placeholders. Result:

| field | official | our `2p8t` guess | verdict |
|---|---|---|---|
| `hidden_size` | **7168** | 7168 | correct |
| `num_hidden_layers` | **93** | 61 | wrong |
| `num_attention_heads` | **96** | 64 | wrong |
| `head_dim` (KDA) | **128** | derived 112 | wrong |
| `num_experts` | **896** | 896 | correct |
| `num_experts_per_token` | **16** | 16 | correct |
| `num_shared_experts` | **2** | 1 | wrong |
| `moe_intermediate_size` | **3072** | 2048 | wrong |
| `routed_expert_hidden_size` | **3584** | (no such concept) | **structural gap** |
| `intermediate_size` (dense L0) | **33792** | derived | wrong |
| `q_lora_rank` | **1536** | asserted `None` | **hard blocker** |
| `kv_lora_rank` | **512** | 512 | correct |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | **128 / 64 / 128** | 128 / 64 / 128 | correct |
| `vocab_size` | **163840** | 163840 | correct |
| `max_position_embeddings` | **1048576** | 4096 | wrong (1M) |
| `hidden_act` | **`situ`** (beta 4.0, linear_beta 25.0) | silu | **gap -- FIXED, see sec 2** |
| `first_k_dense_replace` | **1** | 1 | correct |
| `rms_norm_eps` | **1e-5** | 1e-5 | correct |
| `tie_word_embeddings` | **false** | false | correct |
| router | `sigmoid`, `noaux_tc`, `moe_renormalize: true`, `routed_scaling_factor: 1.0`, `use_grouped_topk`, groups 1 | sigmoid + renorm | mostly correct |

**Layer pattern.** Official `full_attn_layers` is
`[4,8,...,88,92,93]` -- every 4th layer **plus layer 93**, so the last two
layers (92, 93) are BOTH full attention: 24 full / 69 KDA. Our
`_alternating_kda_mla_layers` puts every 4th layer on MLA and would assign
layer 93 to KDA (93 % 4 == 1). **Off-by-one at the tail; must special-case the
final layer.**

## 2. FIXED already: SiTU (fork `8c123baa`)

```
situ(g) = beta * tanh(g / beta) * sigmoid(g)
up      = linear_beta * tanh(up / linear_beta)
out     = situ(gate) * up
```

A soft-clipped SiLU: tanh caps magnitude at +/-beta, sigmoid keeps the SiLU
shape near 0; the linear branch is clipped the same way. Gated over BOTH
branches, so it replaces the `act_fn(gate) * up` product rather than acting
elementwise. Implemented with the official betas as defaults, 9 CPU tests
(exact formula, both saturation caps, `linear_beta=None`, dtype, MLP
composition, silu unchanged). Default path bit-identical: fsdp8 debugmodel
step-1 still 7.58611 / 1.8625.

## 3. Structural gaps, in priority order

### 3.1 Stable LatentMoE -- the biggest one

Official routed-expert path (from the modeling code):

```
routed_expert_down_proj : hidden 7168 -> moe_hidden 3584     (shared)
  per-expert KimiBlockSparseMLP(hidden=3584, intermediate=3072)
      w1, w3 : 3584 -> 3072      w2 : 3072 -> 3584
routed_expert_norm      : RMSNorm(3584)                      (latent_moe_use_norm)
routed_expert_up_proj   : moe_hidden 3584 -> hidden 7168      (shared)
y = routed + shared_experts(identity)
```

So **the routed experts live in a compressed latent space**, entered and left
through two shared projections, with an RMSNorm on the latent before the
up-projection. That is what "Latent" means, and the norm plus the reduced
routing dimension are what the blog calls "Stable". Our MoE runs experts
directly on `hidden_size` with no latent bottleneck and no shared
down/up-projection -- a real structural difference, not a hyperparameter.

Consequence for us beyond parity: the expert weights are `3584x3072` instead of
`7168xN`, which changes EP sharding shapes, the state-dict adapter, and the
MXFP4 packing layout.

### 3.2 MLA: q-LoRA + output gate

- `q_lora_rank: 1536` -> `q_a_proj (7168->1536)`, `q_a_layernorm`,
  `q_b_proj (1536->heads*qk_head_dim)`. **Our `KimiMLAAttention` asserts
  `q_lora_rank is None`** ("This port targets Kimi 48B-A3B's q_lora_rank=null
  path"), so the official config cannot even be constructed. Hard blocker for
  loading official weights.
- `mla_use_output_gate: true` -> `g = g_proj(hidden).sigmoid()`, then
  `attn_output = attn_output * g`. We ship a provisional gate; the form to match
  is a sigmoid on a full `hidden -> projection_size` projection applied to the
  attention output.
- `mla_use_nope: true` matches our NoPE variant.

### 3.3 KDA: full-rank gate + gate lower bound

`use_full_rank_gate: true` gives the gate its own full
`nn.Linear(hidden_size, projection_size)` rather than fla-core's default
low-rank gate, and `gate_lower_bound: -5.0` is passed into the kernel as
`safe_gate=True, lower_bound=-5.0`. Needs an fla-core capability check before
we can claim parity.

### 3.4 Vision tower (new surface for us)

Official `vision_config`: a 27-layer ViT, `vt_hidden_size 1024`,
`vt_intermediate_size 4096`, `vt_num_attention_heads 12`,
`qkv_hidden_size 1536`, `patch_size 14`, RMSNorm, `gelu_pytorch_tanh`,
`mlp_type mlp2`, `attn_bias false`. Projector: `mm_projector_type
patchmergerv2`, `merge_kernel_size [2,2]`, `merge_type sd2_tpool`,
`mm_hidden_size 1024 -> text_hidden_size 7168`. Position embeddings:
`pos_emb_type divided_fixed` with `init_pos_emb_height/width 64` and
**`init_pos_emb_time 4`** (a time axis -- video, not just images), bilinear
interpolation. Entry point: `<|kimi_image_placeholder|>`, token id 163605,
top-level class `KimiK3ForConditionalGeneration`.

Our `multimodal_model.py` is LLaVA-style scaffolding with a pluggable vision
module and no concrete tower -- so the layout is compatible in spirit, but every
concrete choice above (patchmergerv2, sd2_tpool merge, divided_fixed 3D pos
emb) is new work.

## 4. Deliberate divergences (NOT bugs; keep and document)

- **Pseudo-query zero-init.** Official inits with std=0.02; we zero-init so a
  graft onto pretrained weights is an exact no-op at step 0
  (adapter-correctness anchor). Correct for grafting, wrong for from-scratch --
  both should be selectable, defaulting to official for a K3-faithful flavor.
- **PP cross-stage adapter.** Nothing in the official release addresses
  multi-node pipeline parallelism for AttnRes; our adapter remains our own
  contribution.
- **`<2% overhead`.** Still to be checked against the report's definition
  (algorithm FLOPs), and still not the same quantity as our +2.7% PP-adapter
  step-time on PCIe. Do not conflate.

## 5. Quantization: our scope is wrong

Official `quantization_config`: `mxfp4-pack-quantized`, `group_size 32`,
`num_bits 4`, symmetric, `scale_dtype torch.uint8`, `strategy group`,
`observer minmax`, `dynamic false`, `quant_method compressed-tensors`,
`quantization_status compressed`, and `input_activations: null`.

The **ignore** list is the load-bearing part:

```
re:.*self_attn.*            re:.*shared_experts.*
re:.*mlp\.(gate|up|gate_up|down)_proj.*
re:.*lm_head.*              re:.*vision_tower.*     re:.*mm_projector.*
```

So in the shipped checkpoint **only the routed experts are MXFP4**; attention,
shared experts, the dense FFN, lm_head, and the whole vision stack stay bf16.
Our QLoRA path quantizes a much broader set of LoRA base layers (q_proj,
kv_b_proj, o_proj, gate/up/down, shared-expert w1/w2/w3 -- i.e. exactly several
things K3 leaves in bf16). Matching K3 means the packing scope becomes
"routed experts only".

Two further notes:

- `input_activations: null` means the **checkpoint** is weight-only MXFP4. The
  blog's "MXFP4 weights with MXFP8 activations" is a training/serving-time
  scheme (QAT from SFT onward), not a checkpoint field -- our `mxfp4_qat.py`
  emulation is the right place for the activation half, and it needs an MXFP8
  activation path to match.
- `scale_dtype: torch.uint8` matches what we already do (we store e8m0 scales
  viewed as uint8 because FSDP2 has no float8_e8m0fnu all-gather kernel).

## 6. Work queue (execute in this order)

1. **DONE** SiTU (`8c123baa`).
2. `q_lora_rank` support in `KimiMLAAttention` -- removes the assert that blocks
   the official config outright.
3. MLA output-gate form check against `g_proj(h).sigmoid()`.
4. Regenerate the K3 flavor from `official_k3/config.json` (93 layers, 7168, 96
   heads, 896/16/2, 3072/3584, 1M positions, the 92+93 full-attn tail) and
   delete the provisional `2p8t` guesses.
5. Stable LatentMoE (down/up projections + latent RMSNorm) -- structural.
6. KDA full-rank gate + `gate_lower_bound`, gated on an fla-core capability
   check.
7. Quantization scope -> routed experts only; MXFP8 activation path in QAT.
8. Vision tower + patchmergerv2 projector + divided_fixed 3D pos emb.
9. Re-run the whole parallelism matrix on whatever survives, and re-check the
   report's overhead definition.
