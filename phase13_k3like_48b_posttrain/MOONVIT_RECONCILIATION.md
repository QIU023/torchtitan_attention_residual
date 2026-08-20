# MoonViT-V2:报告与 checkpoint 不一致处的取舍

从 `kimi_k3/moonvit.py` 的模块 docstring 搬出。原文 50 行。内容未改。

matters here, because the two disagree.

Report sec 2.4 says "attention is factorized into intra-frame spatial and
inter-frame temporal passes". The shipped tower does NOT do that: each block has
exactly one ``wqkv`` and one ``wo`` (27 of each in
``model.safetensors.index.json``), and ``MoonViTEncoderLayer.forward`` runs a
single varlen attention whose ``cu_seqlens`` spans a sample's whole ``t*h*w``
token set. So frames interact through one joint 3-D attention, not two passes.
An earlier version of this file implemented the factorized reading and matched
the reported 0.4B parameter count -- which proves only that parameter count
cannot distinguish "one projection set used twice" from "one projection set used
once".

Layout, from the checkpoint keys::

    vision_tower.patch_embed.proj.weight            Conv2d, no bias
    vision_tower.patch_embed.pos_emb.weight         learned 2-D spatial table
    vision_tower.encoder.blocks.{i}.norm0.weight    27x, RMSNorm
    vision_tower.encoder.blocks.{i}.wqkv.weight     27x, no bias
    vision_tower.encoder.blocks.{i}.wo.weight       27x, no bias
    vision_tower.encoder.blocks.{i}.norm1.weight    27x
    vision_tower.encoder.blocks.{i}.mlp.fc0.weight  27x
    vision_tower.encoder.blocks.{i}.mlp.fc1.weight  27x
    vision_tower.encoder.final_layernorm.weight
    mm_projector.proj.{0,2}.weight                  2 Linears, no bias
    mm_projector.post_norm.weight                   RMSNorm AFTER the projection

Two details the key list settles that the config alone does not:

* There is no ``pos_emb.time_weight`` key. The time component is a FIXED 1-D
  sincos table registered as a non-persistent buffer -- that is what the
  "fixed" in ``pos_emb_type: divided_fixed`` refers to. Only the 2-D spatial
  table is learned, and it is interpolated to the input's patch grid.
* ``mm_projector`` has ``post_norm`` and no pre-norm, matching
  ``PatchMergerMLPV2`` rather than ``PatchMergerMLP``.

Positional information is therefore carried twice: the absolute divided_fixed
embedding added at the patch embed, AND 2-D RoPE applied to q/k inside every
block.

Shape suffixes: L packed tokens across the whole batch, D model dim
(vt_hidden_size), Q qkv_hidden_size, A heads, K head_dim, C text_hidden_size.
Inputs are NaViT-style packed -- a flat ``(L, ...)`` token stream plus a
``grid_thws`` table of ``(t, h, w)`` per sample -- so one batch can mix
resolutions and frame counts, which is what native-resolution training needs.
