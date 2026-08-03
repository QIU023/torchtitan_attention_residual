# A multimodal mini checkpoint that is actually multimodal

## Why the previous one was not

`/workspace/k3mini_hf` started as a hand-written K3 config skeleton with no
weights, declaring `architectures: [KimiK3ForConditionalGeneration]` -- K3's real
identity. The 946 tensors in it later came from converting a torchtitan seed
checkpoint generated with `kimi_linear_k3mini_block_attn_res`, a TEXT-ONLY
flavor. So it claimed to be multimodal while containing zero vision tensors, and
vLLM's multimodal path failed on the mismatch.

Two ways out were available and one of them was wrong. Repointing `auto_map` at
the multimodal config makes the config self-consistent but the weights still have
no tower. Changing `architectures` to the text-only class makes it load -- by
demoting K3 to Kimi-Linear and discarding the fact that K3 is natively
multimodal. That is the one I started on and it was the wrong trade.

## What was built instead

`kimi_linear_k3mini_vl`: the text k3mini plus a SHRUNK MoonViT-V2. Sizes matter
here -- the released tower is 447.4M against k3mini's 80.9M text side, 5.5x, so
a debug run would measure the encoder rather than the K3 structure under test.
At 4 layers / hidden 256 / 4 heads (head_dim 64, as released) the tower is 6.44M,
8% of the text side.

Shrunk, not simplified. Every structural feature of MoonViT-V2 is kept: the
single varlen attention pass (not the factorized one the report describes), 2D
RoPE with the divided_fixed absolute embedding, sd2_tpool, PatchMergerMLPV2. So
NaViT packing, the projector and the image_mask splice are all genuinely
exercised.

Uses `KimiK3MultimodalConfig`, the release-faithful one: the projector belongs to
the tower (`mm_projector` is a MoonViT child in the checkpoint) and the tower is
NOT frozen, since report sec 2.4 trains MoonViT-V2 from scratch jointly with the
text model.

## State

Trains under FSDP at 87,376,316 parameters -- loss 7.71686 / 7.66182 / 7.51760
over three steps, and 7.71686 / 7.62315 with checkpointing on.

The saved checkpoint is multimodal by inspection: 9,888 keys, of which 390 are
`vision_tower.*` and 39 are projector, with top-level prefixes `language_model`
and `vision_tower`.

Single-GPU remains blocked by the KDA shared-memory limit that affects every
flavor on this box, so the seed-checkpoint path (single-process only) cannot be
used; the checkpoint above comes from a short FSDP run instead.

## Exported, and the three now agree

`export_vl_to_hf.py` materializes the DCP model tensors, writes safetensors, and
emits a config with the text fields nested under `text_config`, a `vision_config`
matching the shrunk tower, and `auto_map.AutoConfig` pointing at the MULTIMODAL
`KimiK3Config` -- the last of which is what previously made transformers resolve
the text-only config and hand vLLM a flat one.

    /workspace/k3mini_vl_hf: 642 tensors, 30 of them vision_tower.*
    transformers resolves KimiK3Config, text hidden 512, vision layers 4

The exporter refuses to write if the checkpoint has no vision tensors, so it
cannot silently reproduce the inconsistency it exists to fix.

One number to read correctly: the DCP metadata has 390 keys containing
`vision_tower`, but 360 of those are optimizer state. The model-side tower is 30
tensors, which is what a 4-layer MoonViT has, and what the export contains.

Next: retry the vLLM rollout against this.
