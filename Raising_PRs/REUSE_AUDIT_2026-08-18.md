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

## Direction correction

This file was written as "align our code to theirs". That is the wrong frame. What we are
doing is writing every parallelism axis on top of their bare eager model, so:

* **where their interface cannot express what a parallelism needs, the fix goes UPSTREAM**,
  as its own PR, not into a local workaround;
* **only our own unjustified duplication needs cleaning** -- redundant inheritance, classes
  that repeat something `models/common` already has.

Their naming and index conventions are not the target. Two items below moved sides once that
was clear.

## Upstream interface changes the parallelism work needs

Each is a small upstream PR that stands on its own and makes our axis PR smaller.

**1. `GroupedExperts` hardcodes its activation.** `F.silu` is fixed in the forward, so every
model whose gate is not SiLU carries a subclass. Three models already do: `gpt_oss` (clamped
SwiGLU), `kimi_k3_up` -- PR-4025 itself -- and ours (SiTU-GLU, report Eq. 12). Parameterizing
it deletes all three. Independent of PR-4025, since it touches only `models/common`, so it can
be filed now rather than after the merge. **Kitted as PR27** (`gate_up_combine` hook).

**1b. `MoE.forward` feeds the router and the experts the same tensor.** Latent-expert designs
route on the full-width token and dispatch a narrower projection, so they need the two to
differ. Our tree adds a keyword-only `router_input_BLD`; keyword-only because
`LocalMapConfig.in_grad_placements` is a tuple ordered by `_cache_pos_arg_names()`, which a
positional parameter would extend.

The half that is easy to miss is in `moe_sharding.py`, not `moe.py`: `in_src_shardings` /
`in_dst_shardings` are keyed by parameter NAME, and `_redistribute_inputs` silently skips a
name it does not find. So an unnamed second input reaches the router unredistributed -- a
Replicate activation arriving at a gate that declares SP, with nothing raised. We were
patching around this with `dataclasses.replace` at model-build time. **Kitted as PR28**,
independent of both PR-4025 and PR27.

**2. `ShardingConfig` cannot express CP or PP.** Its six placement fields
(`state_shardings`, `in_src`/`in_dst`, `out_src`/`out_dst`, `local_map`,
`in_grad_placements`) say how a module's parameters and IO are sharded on a mesh. They cannot
say "this axis splits the SEQUENCE" (CP) or "this module is split across stages" (PP), because
neither is a placement on the module's own tensors. That is why our CP and PP are imperative
while TP and EP can be declarative, and the biggest single reason our `parallelize.py` is 1656
lines against their 99.

Worth raising as a design question BEFORE the axis PRs, because the answer decides whether
CP/PP ship imperative as written or get rewritten declarative. Ask on the RFC rather than
guessing locally.

**3. EP transport needs nothing.** `BaseEPTokenDispatcher` is an ABC with exactly `dispatch`
and `combine`, `wire_meshes` installs the mesh, `init_buffer` is the persistent-buffer hook,
and `deep_ep.*` is already an optional import in `pyproject.toml`. MoonEP needs one subclass
and one entry in that list. Checked -- see `kimi_k3/moon_ep_dispatcher.py`.

## Our own duplication: one real item, and a claim of mine that was wrong

An earlier draft of this file said three of our files subclass `GroupedExperts` for three
different reasons, and used it as the strongest argument for the upstream change. That came
from reading `grep -l` output as inheritance. Actually:

| file | what it really does |
| --- | --- |
| `moe.py` | `KimiSiTUGroupedExperts(GroupedExperts)` -- the one real subclass |
| `mxfp4_qat.py` | `MXFP4QATGroupedExperts(parent_cls)` -- dynamic factory, parent passed in |
| `quant_scope.py` | `isinstance(module, GroupedExperts)` -- a type test |
| `lora.py` | mentions it in comments only |

So we subclass it ONCE. The upstream case survives on three separate MODELS doing it, not on
the three-files-of-ours version I wrote.

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
| `quant_scope.py` | does NOT subclass -- `isinstance` test | nothing to do |
| `lora.py` | does NOT subclass -- comments only | nothing to do |
| `vit_cp_plan.py` (pure planning) | nothing comparable | keep; it exists so scheduling is testable without ranks |
| `layout.py` (block propagation tables) | nothing comparable | keep |
| `knobs.py` (config-first topology) | `torchtitan/config` for job config | keep, but it is model-local by design; do not upstream |

Corrected above: we subclass it ONCE. The upstream case rests on three separate models doing
it (gpt_oss, PR-4025, ours) -- still not one model's quirk, but not the version I wrote.

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

**Naming and index base are not alignment targets** (see the direction correction above).
The 63 index sites matter only where our code meets theirs -- the state-dict adapter and the
config fields a user sets -- not across our whole tree. Where they do meet:
`state_dict_adapter` already carried a bug from exactly this confusion (`kda_layers_zero_based`
used where `is_kda_layer` was meant, disagreeing on layer 0). Do it in one commit, before the
diffs are regenerated against their merge, or the ambiguity propagates into five of them.
