"""Scratch flavors for the attention residual measurement (never committed).

ATTNRES_DIM, ATTNRES_LAYERS, ATTNRES_BLOCK (env) size the wide flavor; the
debug flavor is the stock one. Both train the language model only (no images),
AdamW, activation checkpointing as the recipe has it.
"""
import os
from dataclasses import replace

from torchtitan.trainer import Trainer


def _base() -> Trainer.Config:
    from torchtitan.components.optimizer import default_adamw
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    config.optimizer = default_adamw(lr=8e-4)
    return config


def attnres_debug() -> Trainer.Config:
    return _base()


def attnres_wide() -> Trainer.Config:
    """The debug flavor's layout at a wider hidden size and a deeper stack."""
    from torchtitan.models.kimi_k3 import _kimi_k3_config

    dim = int(os.environ.get("ATTNRES_DIM", "4096"))
    num_layers = int(os.environ.get("ATTNRES_LAYERS", "16"))
    block = int(os.environ.get("ATTNRES_BLOCK", "2"))
    config = _base()
    stock = config.model
    config.model = _kimi_k3_config(
        max_context_length=stock.max_context_length,
        dim=dim,
        moe_comm_backend="standard",
        vocab_size=163840,
        num_layers=num_layers,
        full_attention_layers={i for i in range(3, num_layers, 4)},
        attn_res_block_size=block,
        num_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=4 * dim,
        latent_dim=1024,
        expert_hidden_dim=1024,
        num_experts=32,
        top_k=4,
        num_shared_experts=2,
        vision_encoder=None,
        attn_backend="flex",
    )
    return config
