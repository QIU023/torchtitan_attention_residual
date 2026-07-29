# veRL actor: how far it gets, and the one thing blocking it

## Working

veRL 0.8.0 ships a torchtitan engine (`verl/workers/engine/torchtitan/`), which
settles the question standing in CLAUDE.md since the handoff: the titan backend
exists, so our model can be veRL's actor with no HF wrapper and no Megatron.

Reached, against real veRL machinery and a real HF config dir:

    [VERL] seed checkpoint: /workspace/overnight2/seed_ckpt/checkpoint/step-0
    [VERL] hf_config: type=kimi_k3 hidden=512 layers=21 vocab=163840
    [VERL] derived name=kimi_k3 flavor=kimi_linear_k3mini_block_attn_res
    [VERL] engine constructed: TorchTitanEngineWithLMHead

Three of the four upstream gaps are already fixed in the QIU023/verl fork
(model_type map, experiments/ namespace fallback, the param_groups optimizer API);
this session added the `kimi_k3` model_type entry. The remaining upstream item is
that `peft` is imported eagerly by verl's package init and needs
`BloomPreTrainedModel`, gone in transformers 5.x -- the titan engine itself does
not use peft, so the smoke stubs it.

## The blocker: it wants HF safetensors, and our vocab does not line up

The engine does not load the DCP seed. It loads HF-format weights from
`model_config.path`, and the error names HF keys:

    Missing key in checkpoint state_dict:
      language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight

`CheckpointConfig` exposes no load-path field to redirect (measured: none of
load_path / initial_load_path / initial_load_model_only exist on it), so the fix
is to put real safetensors at that path. torchtitan ships
`scripts/checkpoint_conversion/convert_to_hf.py`, which also exercises our own
`to_hf` adapter -- the right tool. It needs the same
`torchtitan.models.kimi_k3` alias veRL needed.

It then fails on the vocab:

    ValueError: Size mismatch between saved torch.Size([2016, 512])
                and current: torch.Size([163840, 512]) for embed_tokens.weight

This is the "single deliberate deviation" recorded when k3mini was created,
surfacing as a real blocker. `model_registry("kimi_linear_k3mini_block_attn_res")`
builds vocab 163840 (K3's own), while the TRAINER config overrides it to 2016 so
the flavor can use the repo's bundled tokenizer without downloading assets. The
seed checkpoint therefore has a 2016-row embedding that cannot load into the
model the registry builds -- and veRL resolves the flavor through the registry, so
it gets the 163840 version.

Two ways out, both small:

1. give k3mini a real 163840-entry tokenizer, so the trainer stops needing the
   override -- clean, but adds an asset download to a flavor that was designed to
   run without one;
2. make the flavor's `model_registry` path honour the same 2016 override the
   trainer config applies, so both sides agree -- keeps k3mini self-contained,
   but means the registry no longer reports K3's true vocab, which is exactly
   what veRL matches flavors on (dim / n_layers / vocab_size), so the shape match
   would have to change with it.

(1) is the honest one for anything downstream of weight loading. (2) is faster if
the goal is only to smoke the actor.
