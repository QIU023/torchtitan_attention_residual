# Scaling-law 配置表的取值依据

从 `kimi_k3/model_configs.py` 的模块 docstring 搬出。原文 34 行。内容未改。

multi-node). Each ``_build_config`` call returns a tuple of
``(kimi_config, num_blocks)`` so a caller can wire it into either
:class:`KimiK3Model` (baseline, ``num_blocks=None``) or
:class:`KimiK3AttnResModel` (AttnRes variant, ``num_blocks=N``).

The paper's Table 2 fields ``d_model``, ``d_ff``, ``L_b`` (= number of
decoder layers), ``lr`` and ``batch_size`` are preserved verbatim. The
vocab / MoE / KDA / MLA knobs default to the 48B-A3B reference config
shape, just scaled by ``d_model``. Specifically:

* Vocab = 163840 (Kimi's native tokenizer, tied in scaling-law to
  keep the non-embedding activated param count matching the paper).
* MoE: ``num_experts_per_token=8``, ``num_shared_experts=1``,
  ``moe_intermediate_size = d_ff``, ``first_k_dense_replace=1``.
* MLA (on ``full_attn_layers``): ``qk_nope_head_dim=128``,
  ``qk_rope_head_dim=64``, ``v_head_dim=128`` scaled to fit
  ``d_model/num_heads``.
* KDA (on ``kda_layers``): head_dim scaled so
  ``num_heads × head_dim ≈ d_model``.
* KDA:MLA = 3:1 ratio matching 48B-A3B pattern (every 4th layer is MLA).

The :attr:`scaling_law_sizes` dict maps size-name → Python constructor;
callers pass a ``num_blocks`` kwarg to pick the AttnRes variant.

These builders return ``(kimi_config, num_blocks)`` tuples for direct
model construction (CPU tests, ad-hoc experiments). The torchtitan
integration lives elsewhere: ``KimiK3Spec`` in ``model.py`` is the
``BaseModel.Config`` shim, and ``config_registry.py`` holds the
``Trainer.Config`` flavors the ConfigManager resolves by name.
