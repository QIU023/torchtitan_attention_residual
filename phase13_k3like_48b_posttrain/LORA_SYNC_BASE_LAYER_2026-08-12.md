# Adapter-only weight sync: the rollout decides which projections are wrapped

The adapter-only GRPO arm reached the right sequence and then died in vLLM:

    weight sync: adapter mode, base_sync_done=False, shipping 706 tensors (rank 16, 139 wrapped projections)
    weight sync: adapter mode, base_sync_done=True,  shipping 278 tensors (rank 16, 139 wrapped projections)
    ...
    KeyError: 'layers.0.self_attn.q_proj.weight'

So the `peft_config` fix worked -- the two-phase base-then-adapter sequence engaged, and
278 = 139 x 2 confirms the adapter half is naming what it should. What was wrong was the
BASE half's naming, and it was wrong on a premise rather than in a detail.

## The premise that was wrong

The base half renames a wrapped projection's key to PEFT's `base_layer` form, because a
LoRA-enabled rollout holds the base weight one level deeper. The first implementation
derived that set from the TRAINER's wrappers -- the 139 modules `apply_lora` actually
wrapped.

vLLM does not use that set. `get_supported_lora_modules` walks `named_modules()` and
collects leaf names by module TYPE:

    """
    In vLLM, all linear layers support LoRA.
    """

so enabling LoRA wraps EVERY linear, whatever the adapter's `target_modules` say.
`_create_lora_modules` then gates only on that leaf-name match. K3 diverges from the
trainer's set on the KDA subtree, which `apply_lora` skips structurally -- so a KDA
`q_proj` is unwrapped on the trainer side and wrapped on the rollout side, and the base
half shipped a plain name into a `params_dict` that only had `base_layer`.

verl already knew this, in a comment on the utility the megatron engine uses:

    # vLLM needs to target all-linear no matter about specific LoRA config

## Why it surfaced 100 lines from its cause

vLLM maps a source projection onto its packed destination and then checks:

    name_mapped = name.replace(weight_name, param_name)
    # Packed projections are only present on compatible layers.
    if name_mapped not in params_dict:
        continue

A LoRA-wrapped destination is therefore read as "this layer has no packed projection",
falls out of the stacked loop, and lands in the plain-name fallback -- where the KeyError
names the SOURCE projection and nothing in the message points at LoRA. That guard is
correct for its purpose (K3 mixes KDA and MLA layers, so packed projections genuinely
exist on only some layers); it just cannot distinguish "wrong layer type" from "wrapped".

## The fix

Delegate to verl's own `add_base_layer_suffix`, which renames by name suffix for exactly
this reason, plus the K3 projections missing from its shared `STACKED_PARAMS`:

| added suffix | vLLM destination |
|---|---|
| `.b_proj` `.f_a_proj` `.g_proj` | fused into `in_proj_qkvgfab` (KDA front end) |
| `.f_b_proj` | its own linear |
| `.q_conv1d` `.k_conv1d` `.v_conv1d` | `self.conv1d = ColumnParallelLinear(...)` -- see below |
| `.self_attention_res_proj` `.output_attn_res_proj` `.mlp_res_proj` | Block AttnRes graft |
| `.routed_expert_down_proj` `.routed_expert_up_proj` | latent MoE W_down / W_up, `ReplicatedLinear` |
| `.block_sparse_moe.gate.weight` + `.e_score_correction_bias` | `GateLinear(ReplicatedLinear)`; the shared list spells this `.mlp.gate.` |
| `.w1` `.w2` `.w3` | per-expert, for a different reason -- see below |

The rename goes on the SOURCE name, and it is only sound because vLLM's mapping is a
substring replace on the module segment: `.q_proj.base_layer.weight` still resolves to
`.in_proj_qkvgfab.base_layer.weight`. That composition is pinned by a test rather than
left as a reading of vLLM's loader.

What is NOT renamed: norms, layernorms, `A_log`, `dt_bias`, and the embeddings plus
`lm_head` -- those last two only because K3 declares no `embedding_modules`, which is why
verl's shared list adds `.embed_tokens.weight` for llama and not here.

### Two of these are not readable off the name

**The KDA short convolution is a linear.** `kda.py` builds it as
`self.conv1d = ColumnParallelLinear(...)` and unsqueezes the weight into conv layout
afterwards, so LoRA wraps it like any projection. It was excluded in the first version of
this fix on the reasoning that a conv is not a `LinearBase`, and a test was written
asserting it keeps its plain name. The rule that survives: **ask what TYPE the vLLM
destination is, not what the name looks like.**

**The per-expert weights are renamed even though nothing wraps them.** They land in
FusedMoE's stacked `w13` / `w2` params, but the expert mapping builds its weight_name as

    f"experts.{logical_id}.{weight_name}.{lora_base_layer_prefix}"

with the prefix set from `any(".base_layer." in n for n, _ in model.named_parameters())`.
So **one LoRA-wrapped module anywhere in the model changes what the expert loader expects
every expert key to be called** -- and an un-suffixed expert key matches no mapping entry
at all, rather than mismatching one, so it falls through to the plain-name lookup and
fails there. This document's first version had it exactly backwards, listing the experts
as out of scope because they travel through the expert mapping; the expert mapping is what
demands the suffix.

## What this retires

The previous commit turned a silent skip in the exact-fqn helper into a warning naming
both sides, on the theory that the KeyError came from a projection whose base key was
absent. It could not fire here: an `hf_names` entry that is PRESENT but names a projection
the rollout wrapped differently is not a missing key. The helper and its warning are gone
along with the premise.

## How this was actually converged

Three rounds, each one KeyError further in, which is the method working rather than
failing: the destination misses `params_dict`, vLLM's stacked loop reads that as "packed
projection not present on this layer", and the plain-path fallback names the source key.

| round | failing key | what it taught |
|---|---|---|
| 1 | `layers.0.self_attn.q_proj.weight` | the wrapped set is the ROLLOUT's, not the trainer's |
| 2 | `layers.0.self_attn.q_conv1d.weight` | the short conv is a `ColumnParallelLinear` |
| 3 | `layers.1.block_sparse_moe.experts.4.w1.weight` | the expert mapping demands the suffix too |

After round 2 the classification was derived once from vLLM's module types instead of
iterated: a test now pins all 41 distinct leaf names the k3mini export ships (23 renamed,
12 plain, plus the router and the experts), so the next architecture change fails in CPU
unit tests rather than 40 minutes into a GRPO run.

## Open

* **Only the text stack is exercised.** The multimodal report_arch flavors cannot serve
  here -- vLLM rejects their MLA dimensions with "No valid MLA prefill backend found" --
  so the vision tower's projections have never been through this path.
* **`routed_experts_prefix` defaults to `"routed_experts"`** in the mapping builder while
  K3's loader passes no prefix, so the destination is
  `experts.routed_experts.base_layer.w13_weight`. Whether that param exists under the
  LoRA-wrapped runner is the next thing this path can disagree about.

## Two evidence rules this cost, both mine

**The merged arm only proves LoRA-OFF names.** Rounds 3 and 4 both leaned on "the merged
arm loads this key, so the key is real". It is not: with `merge=true` the rollout serves no
adapter, so its `params_dict` is the unwrapped model. Using it as a general oracle is what
turned a bug in vLLM's mapping into a "contract" the sender was made to satisfy (round 3),
and then made the opposite name look proven (round 4). The only oracle for a LoRA-enabled
`params_dict` is a LoRA-enabled `params_dict` -- which is why the diagnostic lookup, added
after round 4, should have been added after round 1.

**Do not launch the next round before the previous one exits.** Rounds 3, 4 and 5 ran
CONCURRENTLY on the same two GPUs, because `timeout 3000` keeps a driver alive for 50
minutes and each round was launched right after reading the previous round's log. Four
jobs shared GPUs 0 and 1. The KeyError conclusions survive -- a name is in `params_dict`
or it is not, and contention cannot invent a specific missing key -- but nothing
resource-shaped from those runs is usable as evidence. In particular the merged arm's
closing `DataLoader worker ... killed by signal: Killed`, which was waved off as benign
teardown, has exactly the signature of memory pressure and was recorded under four-way
contention.
