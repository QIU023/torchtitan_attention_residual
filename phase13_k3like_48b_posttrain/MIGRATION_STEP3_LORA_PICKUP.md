# LoRA / MXFP4 on the upstream K3 model -- reconnaissance, for pickup

Not started. Everything below is measured on the tree, not planned in the abstract, so
whoever picks this up starts from facts rather than from a design.

## The one structural difference, and it is in our favour

Our `apply_lora` skips the `KimiDeltaAttention` subtree **structurally**, and the reason is
in `lora.py`:

> KDA-internal projections are NOT targetable: `KimiDeltaAttention` reads `linear.weight`
> directly for the fla kernels (module forward is bypassed), so a wrapper there would be
> silently dead.

**Upstream's KDA does not do this.** Its forward calls `self.q_proj(x_BLD)` -- an ordinary
module call -- and so does the KCP forward added in the CP step. So on this tree the KDA
projections ARE targetable and the structural skip does not port. LoRA coverage on the
upstream model is strictly larger than on ours.

Worth confirming with a probe before relying on it: wrap one KDA projection, run a step,
and check the adapter's gradient is non-zero. "Silently dead" is exactly the failure this
whole note is about, and it should not be traded for an assumption in the other direction.

## Name mapping

Linear leaf names on the upstream model, enumerated from a built model rather than read off
the source:

`attention_res_proj, beta, ffn_res_proj, forget_a, forget_b, gate, k_proj, lm_head,
output_gate, output_proj, output_res_proj, q_proj, routed_down, routed_up, v_proj, w1, w2,
w3, wkv_a, wkv_b, wo, wq_a, wq_b`

| ours (`DEFAULT_LORA_TARGETS`) | upstream |
|---|---|
| `q_a_proj` / `q_b_proj` | `wq_a` / `wq_b` |
| `kv_a_proj_with_mqa` / `kv_b_proj` | `wkv_a` / `wkv_b` |
| `o_proj` | `wo` |
| `attn_gate_proj` | `gate` |
| `gate_proj` / `up_proj` / `down_proj` | `w1` / `w3` / `w2` |
| `latent.down` / `latent.up` | `routed_down` / `routed_up` |
| (not targetable for us) | `q_proj`, `k_proj`, `v_proj`, `forget_a`, `forget_b`, `beta`, `output_gate`, `output_proj` -- the KDA set, newly available |

Verify the `w1`/`w2`/`w3` mapping before use. Ours are named by role
(`gate_proj`/`up_proj`/`down_proj`); theirs by position, and the SwiGLU convention is
`w2` = down, `w1`/`w3` = the two up-projections, but which of `w1`/`w3` is the gate should
be read off their `FeedForward` rather than assumed.

## Three Linears that must be excluded

`attention_res_proj`, `ffn_res_proj`, `output_res_proj` -- the AttnRes pseudo-queries.
They are zero-initialized on purpose and the carrier story depends on small deltas
accumulating from exactly zero; this is the same reason they are already filtered out of
FP8 (`filter_fqns` note in `attn_res.py`). A LoRA wrapper there starts the effective weight
at zero as well, so it would not obviously break -- which is what makes it worth an
explicit exclusion rather than trusting that it "probably does not matter".

## Routed experts are absent by construction

They are `GroupedExperts` 3-D parameters, not `nn.Linear`, so no name in the table above
reaches them. They get adapted or quantized through the grouped path -- `quant_scope.py`,
`quantize_grouped_experts_mxfp4`. Check whether upstream's `KimiGroupedExperts` keeps the
same 3-D parameter layout our grouped path expects.

## Suggested shape of the change

`apply_lora` is already name-driven via a `targets` tuple, so the port is a target set plus
one generalization: the structural skip currently hard-imports our `KimiDeltaAttention`, so
make the skipped class a parameter (defaulting to today's behaviour) rather than adding a
second copy of the function. Same minimal-generalization move that `conv_with_halo`'s
`activation` argument used in the CP step.

## Gate

The multimodal LoRA arm of the three-arm matrix, plus the check that adapter gradients on a
KDA projection are non-zero.
