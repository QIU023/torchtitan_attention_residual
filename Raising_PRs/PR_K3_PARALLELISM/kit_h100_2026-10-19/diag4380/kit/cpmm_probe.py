"""Local cells for #4380 (dynamic CP of the Kimi K3 vision tower) and its main baseline; logbook kit only.

cell(): the Kimi K3 debug recipe (main's kimi_k3_debugmodel) as #4639's h100 cells build it. Env:
  CP_MODE      none | allgather (head-tail balancer) | ulysses   (default allgather)
  CP_DEGREE    context parallel degree (2); CP_DP dp_shard (1); CP_TP tensor parallel (1, sets EP = TP)
  CP_MINP      dynamic_cp_min_patches when the tree has it (256 by default; 128 splits cc12m-test's 192-patch images;
               off leaves the tree's default, which is None (dynamic CP off) from c5fdbcca5 on)
  CP_TYPECHECK 0/1 (1, as #4639's cells); CP_STEPS (100); CP_OPT recipe (DistMuon) or adamw; seed 42, deterministic.
  CP_AC        0 drops the recipe's activation checkpointing (typechecking already drops it).
  CP_IMG_PX    resize every image to this square (multiple of 28; upsampling allowed) instead of the recipe's 256-patch cap.
  CP_TOWER     k3 swaps in the released K3 tower (27 layers, dim 1024) projecting to the debug text width.
  CP_SEQ       model and micro-batch context length (default the recipe's 2048).
  CP_MAXP      raise the recipe's 256-patch cap to this many patches (64 per side), keeping native image sizes.
  CP_SEED      model-only seed checkpoint every cell loads (seed() writes it to CP_SEED_OUT).
  CP_PACK      1 packs whole samples into each micro-batch (MMSamplePackingConfig, 8 bins) instead of one padded sample.
"""

import os
from dataclasses import replace

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan_recipes.tests import _set_spmd_typechecking


def _square_resize(height, width, **_):
    side = int(os.environ["CP_IMG_PX"])
    return side, side, 0, 0


def cell():
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        AllGatherKVFlexInnerAttention,
        UlyssesFlexInnerAttention,
    )
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.kda import InnerKDA

    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    e = os.environ
    mode = e.get("CP_MODE", "allgather")
    config = kimi_k3_debugmodel(attn_backend="flex", **({"seq_len": int(e["CP_SEQ"])} if e.get("CP_SEQ") else {}))
    if e.get("CP_TOWER") == "k3":
        from torchtitan.models.kimi_k3.flavors import _vision_encoder_config

        config.model.vision_encoder = _vision_encoder_config(
            text_dim=config.model.dim,
            dim=1024,
            qkv_dim=1536,
            hidden_dim=4096,
            num_layers=27,
            num_heads=12,
            init_pos_emb_height=64,
            init_pos_emb_width=64,
        )
    if e.get("CP_IMG_PX"):
        side = int(e["CP_IMG_PX"]) // 14
        dataset = config.dataloader.dataset
        config.dataloader = replace(
            config.dataloader,
            dataset=replace(
                dataset,
                processor=replace(
                    dataset.processor,
                    resize_fn=_square_resize,
                    max_patches=side * side,
                    max_patches_per_side=side,
                ),
            ),
        )
    if e.get("CP_MAXP"):
        dataset = config.dataloader.dataset
        config.dataloader = replace(
            config.dataloader,
            dataset=replace(
                dataset,
                processor=replace(
                    dataset.processor,
                    max_patches=int(e["CP_MAXP"]),
                    max_patches_per_side=64,
                ),
            ),
        )
    if e.get("CP_PACK") == "1":
        from torchtitan.components.data.multimodal.mm_datasets import MMSamplePackingConfig

        config.dataloader = replace(
            config.dataloader,
            dataset=MMSamplePackingConfig(dataset=config.dataloader.dataset, num_packing_bins=8),
        )
    _set_spmd_typechecking(config, typechecking=e.get("CP_TYPECHECK", "1") == "1")
    config.parallelism.data_parallel_shard_degree = int(e.get("CP_DP", "1"))
    if e.get("CP_OPT", "recipe") == "adamw":
        config.optim.optimizer = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
        )
    tp = int(e.get("CP_TP", "1"))
    if tp > 1:
        config.parallelism.tensor_parallel_degree = tp
        config.parallelism.expert_parallel_degree = tp
    vision = config.model.vision_encoder
    if vision is not None and hasattr(vision, "dynamic_cp_min_patches") and e.get("CP_MINP") != "off":
        vision.dynamic_cp_min_patches = int(e.get("CP_MINP", "256"))
    if mode != "none":
        config.parallelism.context_parallel_degree = int(e.get("CP_DEGREE", "2"))
        if mode == "allgather":
            config.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
            inner = AllGatherKVFlexInnerAttention
        else:
            config.parallelism.context_parallel_load_balancer = None
            inner = UlyssesFlexInnerAttention
        config = apply_transforms(
            config,
            [
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: inner,
                        InnerKDA: ContextParallelInnerKDA,
                    }
                )
            ],
        )
    if e.get("CP_AC", "1") == "0":
        config.activation_remat = None
    config.training.steps = int(e.get("CP_STEPS", "100"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    if e.get("CP_SEED"):
        from torchtitan.components.checkpointer import CheckpointManager

        config.checkpointer = CheckpointManager.Config(
            interval=100000, initial_load_path=e["CP_SEED"], initial_load_model_only=True
        )
    return config


def seed():
    """One-GPU seed checkpoint of the same model (CP_SEED_OUT); cells load it with CP_SEED so every sharding starts equal."""
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(attn_backend="flex")
    config.optim.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    config.checkpointer = CheckpointManager.Config(folder=os.environ["CP_SEED_OUT"])
    config.create_seed_checkpoint = True
    config.training.steps = 1
    config.debug.seed = 42
    return config
