"""Local recipe (never committed): the B200 vit_dep cell on the current debug model widened by
DEPW_DIM / 256 (every width that is a multiple of dim scales, heads by count); records every
rank's per-step peak to $DEPW_OUT/rank<r>.json."""

import atexit
import json
import os

import torch

from torchtitan.trainer import Trainer

_RECORDS: list[dict] = []


def _widened(attn_backend="flex", converters=None, moe_comm_backend="standard", *, enable_sp, seq_len=None):
    from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config

    dim = int(os.environ["DEPW_DIM"])
    s = dim // 256
    assert dim == 256 * s and converters is None
    return _kimi_k3_config(
        max_context_length=seq_len or 2048, dim=dim, enable_sp=enable_sp, moe_comm_backend=moe_comm_backend,
        vocab_size=2048, num_layers=17, full_attention_layers={3, 7, 11, 15, 16}, attn_res_block_size=4,
        num_heads=4 * s, q_lora_rank=128 * s, kv_lora_rank=64 * s, qk_nope_head_dim=64, qk_rope_head_dim=32,
        v_head_dim=64, kda_head_dim=128, conv_kernel_size=4, dense_hidden_dim=512 * s, latent_dim=128 * s,
        expert_hidden_dim=128 * s, num_experts=8, top_k=2, num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim, dim=256 * s, qkv_dim=512 * s, hidden_dim=512 * s, num_layers=2, num_heads=4 * s,
            init_pos_emb_height=32, init_pos_emb_width=32),
        attn_backend=attn_backend)


def _record() -> None:
    _RECORDS.append({
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })


def _install() -> None:
    from torchtitan.observability import metrics

    reset = metrics.DeviceMemoryMonitor.reset_peak_stats

    def reset_after_recording(self):
        _record()
        reset(self)

    metrics.DeviceMemoryMonitor.reset_peak_stats = reset_after_recording

    def dump():
        out = os.environ.get("DEPW_OUT")
        if not out or not torch.cuda.is_available():
            return
        _record()
        rank = int(os.environ.get("RANK", "0"))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
            json.dump({"rank": rank, "records": _RECORDS}, f)

    atexit.register(dump)


def depw_bubble_on() -> Trainer.Config:
    from torchtitan.models.kimi_k3 import config_registry
    from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp4_vp2_vit_dep

    original = config_registry.model_registry
    config_registry.model_registry = lambda flavor, **kw: _widened(**kw)
    try:
        config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    finally:
        config_registry.model_registry = original
    assert config.model.vision_dep.enabled and config.model.vision_dep.bubble
    assert config.model.dim == int(os.environ["DEPW_DIM"])
    _install()
    return config
