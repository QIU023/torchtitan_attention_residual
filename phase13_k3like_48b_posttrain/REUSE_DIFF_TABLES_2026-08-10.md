# Findings 6 and 7: difference tables against the upstream components

The maintainer asked these two questions on the eager PR, close to verbatim: why not
parameterize DeepSeek-V3's MLA instead of forking it, and "should use our fused feed
forward". These are the answers, per row, with a converge-or-justify verdict rather than a
narrative. Compared against `torchtitan/models/deepseek_v3/model.py::Attention` and
`torchtitan/models/common/feed_forward.py::FeedForward` at fork HEAD.

## Finding 6 -- `KimiMLAAttention` vs `deepseek_v3.Attention`

| # | difference | verdict |
|---|---|---|
| 1 | **NoPE.** We assert `mla_use_nope=True` and apply no rotary at all; position information rides the KDA recurrence. Upstream calls `self.rope(q_pe, k_pe, positions)` unconditionally. | **Parameterize theirs.** Making `rope` optional (`rope: RoPE.Config \| None`) is a small, self-contained upstream change and removes the largest single reason we have our own forward. |
| 2 | **Gated MLA**, report Eq. 7: `y = W_o[sigmoid(W_g x) (.) o~]`, `W_g` full rank over `num_heads * v_head_dim`, applied before `o_proj`. Plus a per-head variant (`attn_gate_param="per_head"`, with bias) that exists so a graft onto a gate-less checkpoint is a step-0 no-op. | **Genuine addition.** Nothing upstream has an output gate on attention. Upstream already ships `SigmoidGatedFeedForward`, so the pattern is not foreign to the codebase; the natural upstream shape is a `gate: Linear.Config \| None` on the MLA config. |
| 3 | **Parameter names.** `q_proj`, `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` against upstream `wq`, `wq_a`, `wq_b`, `wkv_a`, `wkv_b`, `wo`. | **Ours is deliberate, and it has a migration cost.** The names are the official Kimi export's, so official checkpoints load with no rename layer. Adopting upstream names is possible -- our state-dict adapter already maps titan to HF -- but it invalidates every DCP checkpoint written so far. State it, do not hide it. |
| 4 | `q_lora_rank` sentinel: `None` for the direct-Q path, upstream uses `0`. | **Converge**, trivially. Pure convention. |
| 5 | **No YaRN mscale.** Upstream scales `softmax_scale` by `mscale` when `max_seq_len > original_seq_len`. We do not. | **Not a real difference.** No RoPE means no YaRN interpolation to compensate for. It falls out of row 1. |
| 6 | ~~k_rot broadcast across heads~~ | **NOT a difference, and the finding was wrong to list it.** Upstream does exactly the same thing: `k_pe.expand(-1, -1, k_nope.size(2), -1)` at `deepseek_v3/model.py:135` against our `k_rot.view(B, 1, T, D).expand(B, H, T, D)`. Same semantics, different spelling. |
| 7 | `num_key_value_heads` / `num_key_value_groups` are read into `__init__` and **never used**. Upstream MLA has no GQA fields at all -- the latent KV is shared by construction. | **Delete ours.** Dead config that reads as if GQA were supported. A small real finding surfaced by writing this table. |
| 8 | `inner_attention` default: ours `ScaledDotProductAttention`, upstream `FlexAttention`. | **Already converged** (finding 58). Both build the shared module; only the default differs, and ours is chosen for MLA's asymmetric head_dim (Q/K 192, V 128), which flash rejects. |
| 9 | **Config style.** We take one `KimiK3Config` dataclass and build the Linears internally with `sharding_config=`; upstream takes per-`Linear.Config` fields and builds what the caller passes. | **The real blocker to literal reuse.** Rows 1, 2 and 4 are each small; this one is why the class cannot simply be subclassed today. Converging it is a mechanical but wide change (every kimi flavor's config construction), and it is the one to cost before promising anything upstream. |

**Summary answer to "why not parameterize theirs":** rows 1, 4 and 8 say we should, row 5
disappears with row 1, row 6 was never a difference, and row 7 is ours to delete. What
genuinely needs new upstream surface is row 2 (an optional output gate). What genuinely
costs work on our side is rows 3 and 9. So the honest position is "converge, in that order"
-- not "our MLA is different".

## Finding 7 -- `KimiMLP` vs `common.FeedForward`

| # | difference | verdict |
|---|---|---|
| 1 | **Names.** `gate_proj` / `up_proj` / `down_proj` against `w1` / `w3` / `w2`. The mapping is exact: `gate_proj = w1`, `up_proj = w3`, `down_proj = w2`. | **Rename, not a fork reason.** Same migration cost as MLA row 3: every DCP checkpoint key changes. |
| 2 | **Activation.** Upstream hardcodes `F.silu` in `forward`. We support `silu`, `gelu` and **`situ`**. | **The one substantive row.** SiTU (report sec 4.1) is `beta*tanh(g/beta)*sigmoid(g)`, and when `situ_linear_beta` is set it ALSO clips the linear branch: `up = lb*tanh(up/lb)`. So it is not an `act_fn` swap -- upstream's `w2(silu(w1(x)) * w3(x))` cannot express a transform applied to `w3`'s output. It is also computed in fp32 and cast back, because a product of two saturating nonlinearities is bf16-sensitive near the caps. |
| 3 | `sharding_config=` passed at construction, versus upstream taking prebuilt `Linear.Config`s. | **Not a difference in capability**, only in which layer expresses it -- same as MLA row 9, and it goes away with that one. |

**Summary answer to "should use our fused feed forward":** yes for `silu`/`gelu`, which is
the dense-FFN and shared-experts path on every non-SiTU flavor. The shape that survives
review is `KimiMLP(FeedForward)` overriding `forward` only for `situ`, with `w1/w2/w3`
names and the state-dict adapter carrying the rename. That is the concrete change; the
blocker is the checkpoint-key migration, not the code.

## Finding 54 -- `apply_fsdp` vs `distributed.fsdp.apply_fsdp_to_decoder`

Assessed rather than converged, and the assessment is the part that was missing. Ours is
182 lines, upstream is 257 and does strictly more; the four ways ours lags are real, and
measurable by reference count inside each function:

| feature | upstream | ours |
|---|---|---|
| `enable_symm_mem` | 3 | 0 |
| `dp_mesh_dims` (full_dtensor flattening) | 9 | 0 |
| `edp_mesh_dims` | 3 | 0 |
| `Shard(1)` refinement | 5 | 0 |
| EP prefetch wiring | 13 | 0 |

**Ours does nothing upstream cannot**, once EP is accounted for -- upstream now takes
`ep_degree` and `edp_mesh` itself. What blocks a literal call is a naming contract, and it
is exactly three items:

| upstream reads | ours has |
|---|---|
| `model.tok_embeddings` | `model.embed_tokens` |
| `model.enable_weight_tying` | absent |
| `block.moe_enabled` / `block.moe.routed_experts.inner_experts` | `block.ffn._moe.routed_experts.inner_experts` |

**The important finding: this does NOT need the checkpoint-key migration that findings 6
and 7 need.** Upstream only ever READS those attributes -- there is no assignment to any
model attribute anywhere in the function -- so read-only properties satisfy it.
A property is not in `_modules`, so `named_parameters()`, `state_dict()` and every FQN stay
exactly as they are, while `fully_shard(model.tok_embeddings)` wraps the same object as
`model.embed_tokens`. Three aliases plus a thin AttnRes-tail delta replaces the fork.

**Not implemented tonight, deliberately.** It changes FSDP wrapping for every cell, so the
grouping and therefore the reduction order can move on all eighteen -- and it would move the
matrix baseline that was just re-established this session, which makes "different grouping"
and "regression" hard to tell apart in the same run. That is a call worth making
deliberately rather than at the end of an overnight. The path above is short and now
unblocked.

## What this table changed

Two corrections to the findings themselves, both found by actually diffing rather than
reading the finding text:

* **Finding 6's "broadcast k_rot" row is not a difference.** Upstream broadcasts too. Any
  PR body citing it would have been wrong in front of the maintainer who wrote that code.
* **A new small finding**: the MLA's GQA fields are dead (MLA row 7).

Neither of the two questions turns out to need a defence of forking. Both resolve to
"converge, and here is the ordered cost". Rows 3 and 9 / row 1 all reduce to one shared
decision -- whether to renumber the checkpoint keys -- which is worth taking once, for both
components, rather than twice.

**And finding 54 is NOT part of that decision**, which was the surprise. It looked like the
same blocker -- upstream spells the same modules differently there too -- but because
`apply_fsdp_to_decoder` only reads, three read-only properties satisfy it without touching a
single FQN. So the TIER 7 work splits in two rather than sharing one gate:

* **54 is unblocked**: aliases plus an AttnRes-tail delta, no checkpoint impact. The cost is
  a matrix run, because FSDP grouping moves.
* **6 and 7 share one decision**: adopt upstream's parameter names and migrate the DCP keys,
  or keep the official-export names and justify the fork on that basis. Worth deciding once
  for both.

That split is the actionable output of these tables. It was not visible from the finding
text, which grouped all three as "reuse".
