"""Local flavors (never committed) for the attention-residual runs on H100: the debug model's
layout at ATTNRES_DIM, ATTNRES_LAYERS in blocks of ATTNRES_BLOCK, full attention every fourth
layer, AdamW, a small vision tower so the multimodal loader's images have somewhere to go."""

import inspect
import os

from torchtitan.trainer import Trainer


def attnres_scaled() -> Trainer.Config:
    from torchtitan.components.optimizer.optimizer import default_adamw
    from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    dim = int(os.environ.get("ATTNRES_DIM", "2048"))
    num_layers = int(os.environ.get("ATTNRES_LAYERS", "24"))
    block = int(os.environ.get("ATTNRES_BLOCK", "12"))
    config = kimi_k3_debugmodel()
    config.optimizer = default_adamw(lr=8e-4)
    sp = (
        {"enable_sp": False}
        if "enable_sp" in inspect.signature(_kimi_k3_config).parameters
        else {}
    )
    config.model = _kimi_k3_config(
        **sp,
        max_context_length=config.model.max_context_length,
        dim=dim,
        vocab_size=2048,
        num_layers=num_layers,
        full_attention_layers={i for i in range(3, num_layers, 4)},
        attn_res_block_size=block,
        num_heads=16,
        q_lora_rank=512,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=2 * dim,
        latent_dim=256,
        expert_hidden_dim=256,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=256,
            qkv_dim=512,
            hidden_dim=512,
            num_layers=2,
            num_heads=4,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend="flex",
    )
    return config
