# HF <-> tt 键映射的规则

从 `kimi_k3/hf_key_map.py` 的模块 docstring 搬出。原文 34 行。内容未改。

Kimi K3 released-checkpoint key names <-> ours.

``state_dict_adapter.py`` targets the Kimi-Linear-48B naming, which K3's release
does not use. Rather than mutate a validated adapter, the K3 translation lives
here and the adapter delegates to it when it sees the K3 layout.

Every pattern below is transcribed from ``model.safetensors.index.json`` of
``moonshotai/Kimi-K3``, and ``test_hf_key_map.py`` asserts coverage in both
directions against that file, so an unhandled key is a test failure rather than
a silently dropped tensor.

Naming differences worth knowing, in rough order of how easy they are to miss:

* Everything text-side sits under ``language_model.model.`` -- the release is
  the multimodal wrapper, so even a text-only load has to strip that.
* Block Attention Residuals ARE in the release, as ``self_attention_res_proj`` /
  ``self_attention_res_norm`` (93 each) plus ``mlp_res_proj`` / ``mlp_res_norm``,
  with the final aggregation at ``model.output_attn_res_proj``. We call the
  per-layer pair ``attention_res_proj`` / ``attention_res_norm`` and the final one
  ``output_res_proj``.
* The MoE layers' module is ``block_sparse_moe``; the single dense layer's is
  ``mlp``. They map onto our ``moe`` and ``feed_forward``.
* Routed experts use ``w1`` / ``w2`` / ``w3`` while the SHARED experts use
  ``gate_proj`` / ``up_proj`` / ``down_proj`` -- the same block uses both
  conventions, so a single global rename gets one of them wrong. w1 is the gate,
  w3 the up, w2 the down (annotated as such in the reference).
* The Gated-MLA output gate is ``g_proj``, the same name KDA uses for its own
  output gate. We call the MLA one ``attn_gate_proj`` because our module also
  supports a per-head graft parameterization the release does not have.
* The router's load-balance bias is ``gate.e_score_correction_bias``, which is a
  BUFFER on our side (``expert_bias_E``), not a parameter.
* Routed-expert weights are MXFP4, stored as ``.weight_packed`` plus
  ``.weight_scale`` rather than ``.weight``. Nothing else in the checkpoint is
  quantized, which is the scope ``quant_scope.py`` encodes.
