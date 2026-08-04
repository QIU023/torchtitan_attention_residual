# Does our 2p8t flavor actually equal the released K3? Table 1, row by row

Built on `torch.device("meta")` -- shapes and `numel()` only, nothing
allocated. Ground truth is report Table 1 plus the released
`official_k3/config.json`.

## The correction that prompted this

An earlier pass reported a 2x discrepancy: "we compute ~105B activated, the
report says A50B". **The report does not say A50B.** Table 1 and the abstract
both say **104.2B activated**. "2.8T-A50B" came from this repo's own
`CLAUDE.md`, which is wrong and is corrected in the same commit as this file.

That is the second time in two days a claim was built on a partial read of a
source we already had. The number was never in the report; it was in our notes.

## Table 1 vs the released config vs ours

| report Table 1 | config.json | ours | |
|---|---|---|---|
| #Layers 93 | 93 | 93 | ok |
| Total Parameters 2.78T | -- | **2.779 T** (meta) | ok |
| Activated Parameters 104.2B | -- | **105.4 B** | ok, ~1% over |
| Hidden Dimension 7,168 | 7168 | 7168 | ok |
| Latent MoE Dimension 3584 (0.5x) | `routed_expert_hidden_size` 3584 | 3584 | ok |
| MoE Hidden per Expert 3,072 | `moe_intermediate_size` 3072 | 3072 | ok |
| Routed Experts 896 | 896 | 896 | ok |
| Experts Active per Token 16 | 16 | 16 | ok |
| Shared Experts 2 | 2 | 2 | ok |
| Attention Heads 96 | 96 | 96 | ok |
| Number of Dense Layers 1 | `first_k_dense_replace` 1 | 1 | ok |
| Vocabulary Size 160K | 163840 | 163840 | ok |
| Training Context 1M | -- | 1048576 | ok |
| Composition 69 KDA + 24 MLA | -- | 69 KDA + 24 MLA, MLA at `[4,...,88,92,93]` | ok |
| Activation SiTU-GLU | -- | SiTU, beta 4/25 | ok |
| **Number of MTP Layers: 1 layer** | **`num_nextn_predict_layers`: 0** | **absent** | **three-way mismatch** |
| **ViT 401M, 27 layers, patch 14, 12 heads** | `vision_config` present | **flavor is text-only** | **gap** |

The 1% activated overshoot is accounting, not structure: 105.4 vs 104.2 is
within what embedding/lm_head tying and shared-expert bucketing move.

Where the parameters are, from the meta build:

    routed experts w1/w2/w3   3 x 907.58 B = 2722.7 B   (98.0%)
    KDA q/k/v/g_proj          4 x 6.08 B    (69 layers x 7168 x 12288)
    o_proj                    8.19 B
    shared experts            12.2 B
    latent up/down            4.7 B

Routed activated = 16/896 x 2722.7 = 48.6 B; dense + attention + shared
= ~56.7 B; total 105.4 B.

## Two real gaps

**1. MTP.** Report Table 1 says 1 MTP layer. The released `config.json` says
`num_nextn_predict_layers: 0`. We implement none. So we match the released
artifact and not the report -- which is the right side to be on for checkpoint
loading, but it means "config-only scale-up reproduces the report's model" is
not true as stated: the report's model has an MTP head and ours cannot express
one. Worth saying plainly rather than leaning on the config.json reading.

**2. The 2.8T flavor is text-only.** `kimi_k3_2p8t_block_attn_res` builds a
`KimiK3Spec`, no `vision_config`. Table 1 lists a 401M ViT at 27 layers, patch
14, 12 heads, and the released config carries a full `vision_config`
(merge_kernel_size 2x2, `sd2_tpool`, `patchmergerv2`, `divided_fixed` position
embeddings, mm_hidden_size 1024). Our MoonViT-V2 implements those features and
our multimodal flavors use them, but there is no 2.8T multimodal flavor, so
the scale-up claim currently covers the text backbone only.

## What can and cannot be claimed

**Can**: the text backbone's structure and every scalar in Table 1 that the
released config also carries match field-for-field, and a meta build reproduces
2.78T total and 104.2B activated to within 1%. The layer composition including
the trailing Gated MLA at 93 is right.

**Cannot**: "config-only scale-up to the released 2.8T model". Two things are
missing -- the MTP layer the report describes, and a 2.8T multimodal flavor
wiring the 401M ViT. Both are additive rather than structural, but neither
exists today.

---

## Gap 2 closed: `kimi_k3_2p8t_vl`

The text-only 2.8T flavor is now joined by a multimodal one wrapping the same
backbone in the released vision tower. `MoonViTConfig`'s defaults already are
the released values, so the flavor passes them through rather than restating
them -- a drift in the defaults surfaces here instead of being hidden by a
duplicate. `vision_token_id` is the released `media_placeholder_token_id`
(163605).

Meta build (shapes only):

| | ours | Table 1 |
|---|---|---|
| total | **2.780 T** | 2.78T |
| activated / token | **105.8 B** | 104.2B |
| vision tower | 447.4 M | 401M |

The vision number needed explaining rather than excusing, and it explains
cleanly:

    encoder      397.0 M
    pos_emb        4.2 M   -> 401.2 M  == Table 1's 401M
    projector     46.1 M   (patchmergerv2, the bridge to the text model)
    total        447.4 M

So Table 1 counts the encoder plus its position embeddings and excludes the mm
projector, which is a bridge rather than part of the tower. 401.2 vs 401 is an
exact match; our tower is the released tower plus that projector.

Remaining gap: MTP. Report Table 1 says one MTP layer, the released
`config.json` says `num_nextn_predict_layers: 0`, and we implement none. We
match the artifact, not the report. That one is still open.
