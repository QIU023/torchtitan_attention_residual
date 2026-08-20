# HF <-> tt 键空间对照

从 `kimi_k3/state_dict_adapter.py` 的模块 docstring 搬出。原文 58 行,
是 reviewer 打开文件第一眼撞上的东西;两条仓库规矩都要求注释精简、WHY 进文档。
内容未改。

``ModelSpec.state_dict_adapter`` so both offline conversion and the
Trainer's ``initial_load_in_hf`` path (and veRL's torchtitan engine,
which sets ``initial_load_in_hf=True``) work.

Key-space notes:

* tt keys are the KimiLinear(AttnRes)Model module tree: ``layers.{i}.
  attention.*`` / ``delta_attention.*`` (per-key names
  already match HF), ``feed_forward.{gate,up,down}_proj`` on dense layers,
  ``moe._moe.{router.gate,expert_bias,experts.w*,shared_experts.w*}``
  on MoE layers, plus the AttnRes extras (``attention_res_proj`` etc. and
  the model-level ``final_attn_res_*``).
* HF checkpoints appear with two MoE prefixes in the wild: the official
  Kimi export style (``mlp.*`` with gate_proj/up_proj/down_proj expert
  linears) and the block-sparse style (``block_sparse_moe.*`` with
  w1/w2/w3 routed + gate/up/down_proj shared). Reading accepts both;
  writing emits the official Kimi-Linear-48B export style
  (``block_sparse_moe.*``; dense layer-0 MLP stays ``mlp.*``).
* KDA ``A_log`` is ``[1, 1, H, 1]`` in HF and ``[H]`` in tt; reading
  reshapes, writing passes the tt shape through (the SGLang overlay
  accepts it -- keep in sync with the overlay if this changes).

Quantized (packed) checkpoints: NOT silently accepted. K3 official
weights are expected to ship packed MXFP4 + scales; until the exact
packing is known (2026-07-27 report), any quantization sidecar key or
sub-byte dtype raises with an explicit message instead of being treated
as an ordinary value.
Which mapper decides a key, and why the order is not symmetric
--------------------------------------------------------------
The hand-written table goes FIRST, because it carries VALUE transforms (the 4-D
``A_log`` reshape) that ``hf_key_map`` does not; delegating ahead of it dropped
those silently, producing shape drift rather than a missing key. ``hf_key_map`` is
the fallback for everything the table does not know -- the K3 layouts: latent MoE
(``moe.latent.*``), the AttnRes tail, gated MLA.

Two keys are exceptions the table DECLINES on purpose, because its answer would
belong to a different layout:

* **shared experts** are layout-dependent. The table only knows Kimi Linear's
  ``moe._moe.shared_experts.w1``, while K3's latent path uses
  ``moe.shared_experts.gate_proj``.
* **``g_proj``** is one release name for two different gates -- KDA's own
  ``g_proj`` and gated MLA's output gate, which is ``attn_gate_proj`` here -- so
  resolving it needs the layer type, which only ``hf_key_map`` has. The table
  returned ``g_proj`` unchanged: a non-None answer that suppressed the fallback and
  left ``attn_gate_proj`` unwritten, so an official gated-MLA load kept its gates
  at random init.

Which layout applies is decided the SAME way ``to_hf`` decides it: keys carrying
the wrapper prefix go to ``hf_key_map``, everything else stays on the table, so a
multimodal export is in K3 naming and a text-only export in Kimi Linear naming.
Mirroring that exactly is what closes the round trip -- asking ``hf_key_map`` for a
text-only model's shared experts returned a K3 path the model does not have, and
returned it SUCCESSFULLY, so no ``UnmappedKey`` fallback could catch it.
