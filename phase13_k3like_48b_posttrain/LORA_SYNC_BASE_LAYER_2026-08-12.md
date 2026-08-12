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
| `.self_attention_res_proj` `.output_attn_res_proj` `.mlp_res_proj` | Block AttnRes graft |
| `.block_sparse_moe.gate.weight` + `.e_score_correction_bias` | `GateLinear(ReplicatedLinear)`; the shared list spells this `.mlp.gate.` |

The rename goes on the SOURCE name, and it is only sound because vLLM's mapping is a
substring replace on the module segment: `.q_proj.base_layer.weight` still resolves to
`.in_proj_qkvgfab.base_layer.weight`. That composition is pinned by a test rather than
left as a reading of vLLM's loader.

What is NOT renamed: norms, layernorms, `A_log`, `dt_bias`, the conv1d weights, the
embeddings, and the routed experts (3-D `GroupedExperts` / FusedMoE, which travel through
the expert mapping and have their own LoRA path).

## What this retires

The previous commit turned a silent skip in the exact-fqn helper into a warning naming
both sides, on the theory that the KeyError came from a projection whose base key was
absent. It could not fire here: an `hf_names` entry that is PRESENT but names a projection
the rollout wrapped differently is not a missing key. The helper and its warning are gone
along with the premise.

## Open, and measurable next

* **The routed experts under a LoRA-enabled rollout are untested.** If `MoERunner` is
  wrapped as `FusedMoEWithLoRA`, the expert names move too, and the expert mapping's
  `params_dict[name]` would fail the same way. The base half's expert keys have not yet
  reached that code, because `q_proj` failed first.
* **Only the text stack is exercised.** The multimodal report_arch flavors cannot serve
  here -- vLLM rejects their MLA dimensions with "No valid MLA prefill backend found" --
  so the vision tower's projections have never been through this path.
