"""Local recipes (never committed): the vit_dep cell at a size where GPU compute, not launches,
sets the step: text dim DEPS_DIM (2048) in DEPS_LAYERS (24) layers, a MoonViT tower of
DEPS_VDIM (1024) x DEPS_VLAYERS (12), DEPS_SEQ (4096) tokens per micro-batch."""

import inspect
import os

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp4_vp2_vit_dep


def _scaled() -> Trainer.Config:
    from torchtitan.components.optimizer.optimizer import default_adamw
    from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config

    dim = int(os.environ.get("DEPS_DIM", "2048"))
    layers = int(os.environ.get("DEPS_LAYERS", "24"))
    vdim = int(os.environ.get("DEPS_VDIM", "1024"))
    vlayers = int(os.environ.get("DEPS_VLAYERS", "12"))
    seq = int(os.environ.get("DEPS_SEQ", "4096"))
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    config.optimizer = default_adamw(lr=8e-4)
    sp = {"enable_sp": False} if "enable_sp" in inspect.signature(_kimi_k3_config).parameters else {}
    config.model = _kimi_k3_config(
        **sp,
        max_context_length=seq,
        dim=dim,
        vocab_size=2048,
        num_layers=layers,
        full_attention_layers={i for i in range(3, layers, 4)},
        attn_res_block_size=4,
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
            dim=vdim,
            qkv_dim=3 * vdim,
            hidden_dim=4 * vdim,
            num_layers=vlayers,
            num_heads=16,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend="flex",
    )
    config.model.vision_dep.enabled = True
    config.model.vision_dep.bubble = True
    config.training.max_context_length = seq
    config.training.num_tokens_per_microbatch_per_dp_rank = seq
    return config


def s_bubble_on() -> Trainer.Config:
    return _scaled()


def s_bubble_off() -> Trainer.Config:
    config = _scaled()
    config.model.vision_dep.bubble = False
    return config


def s_prefetch() -> Trainer.Config:
    config = _scaled()
    config.model.vision_dep.bubble = False
    config.model.vision_dep.prefetch = 1
    return config


def s_off() -> Trainer.Config:
    config = _scaled()
    config.model.vision_dep.enabled = False
    config.model.vision_dep.bubble = False
    return config
