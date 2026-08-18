# Reuse audit: what to fix before a reviewer asks

Rule this follows: align with PR-4025 where they have something, keep everything they lack,
and where we duplicate what `torchtitan` already provides, either reuse it or -- if the
reuse is blocked by an upstream limitation -- file that upstream instead of working around
it locally.

## The five diffs, after regrouping

| diff | files | lines |
| --- | --- | --- |
| `0_shared_parallelize` | `parallelize.py`, `knobs.py` | 1958 |
| `a_tp` | `quant_scope.py` | 106 |
| `b_ep_moe` | `moe.py`, `quantile_balance.py` | 514 |
| `c_cp_kcp_dyncp` | `kcp.py`, `vit_cp_plan.py`, `multimodal_model.py`, `moonvit.py` | 2970 |
| `d_pp_attnres_dep` | `pipeline_adapter.py`, `layout.py`, `attn_res.py`, `attn_res_model.py`, `dep_bubble_{plan,runtime,backward}.py`, `vit_prefetch.py` | 4180 |

Vision dynamic CP now rides with CP and DEP rides with PP, as instructed. 17 files, against
the 7 the previous split covered.

**`parallelize.py` is its own diff, not TP's.** All three of TP, EP and CP apply from inside
it -- `apply_tp_kimi_k3` at 797, the CP block at 199 with `_build_cp_subgroups` at 497,
`apply_ep_kimi_k3` at 1289 -- so calling it "the TP file" was wrong, and giving it to one axis
would have left the other two without their application code. Splitting it into
`tp_plan.py` / `cp_plan.py` / `ep_plan.py` is the real fix and is a refactor that needs a
matrix run; until then it is shared infrastructure that lands first.

## `moe.py`: our writing style is upstream's own, and that is the finding

The question was whether a whole file subclassing `GroupedExperts` to change one activation
is bad practice. Checked rather than agreed with:

* `models/common`'s `GroupedExperts` hardcodes `F.silu`;
* `gpt_oss/moe.py` subclasses it for clamped SwiGLU;
* **`kimi_k3_up/model.py` -- PR-4025 itself -- subclasses it too.**

So the pattern is upstream's, and ours (SiTU-GLU, report Eq. 12) follows the same precedent
as the model PR we are aligning to. The defect is not in our file; it is that
`GroupedExperts` takes no activation parameter, which forces every model with a non-SiLU
gate to carry a subclass.

**Upstream PR to file: make `GroupedExperts` take its activation.** It deletes our
`moe.py`'s reason to exist and gpt_oss's too, and it is a small, self-contained change to
`models/common` -- the kind that is easier to land on its own than inside a model PR. File
it BEFORE the K3 parallelism diffs, so `b_ep_moe` shrinks to just `quantile_balance.py`.

## Other duplications found, with what to do

| ours | already in torchtitan | action |
| --- | --- | --- |
| `moe.py` (SiTU-GLU subclass) | `GroupedExperts` (silu hardcoded), `gpt_oss/moe.py` precedent | upstream PR to parameterize the activation, then delete ours |
| `quant_scope.py` subclassing `GroupedExperts` | same base, for MXFP4 scope | folds into the same upstream PR if the activation hook is a general converter hook |
| `lora.py` subclassing `GroupedExperts` | same base | third caller of the same missing parameter -- strengthens the upstream case |
| `vit_cp_plan.py` (pure planning) | nothing comparable | keep; it exists so scheduling is testable without ranks |
| `layout.py` (block propagation tables) | nothing comparable | keep |
| `knobs.py` (config-first topology) | `torchtitan/config` for job config | keep, but it is model-local by design; do not upstream |

Three of our files subclass `GroupedExperts` for three different reasons. That is the
strongest single argument for the upstream change: it is not one model's quirk.

## Alignment cost against PR-4025, measured

| | ours | theirs |
| --- | --- | --- |
| `model.py` | 2815 lines | 760 |
| `class Kimi*` definitions | 22 | 9 |
| files they do not have | 24 files / 11058 lines | -- |
| layer index base | 1-based, **63 reference sites** | 0-based (`full_attention_layers`) |

The 2055-line gap in `model.py` is not duplication to remove: their 760 lines get KDA, MLA
and MoE from `models/common`, and ours carries the AttnRes graft, MTP, the quantization scope
and the LoRA attachment points -- all things they lack and all of which stay.

**The 63 index sites are the one alignment item that is a correctness risk, not a rename.**
`state_dict_adapter` already carried a bug from exactly this confusion (`kda_layers_zero_based`
used where `is_kda_layer` was meant, disagreeing on layer 0). Do it in one commit, before the
diffs are regenerated against their merge, or the ambiguity propagates into five of them.
