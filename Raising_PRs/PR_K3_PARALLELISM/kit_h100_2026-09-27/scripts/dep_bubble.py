"""Local recipes (never committed): DEP on 4 x H100, from the B200 vit_dep cell."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp4_vp2_vit_dep
from torchtitan_recipes.tests.multimodal import set_rank_conditional_image_presence


def dep_bubble_on() -> Trainer.Config:
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    assert config.model.vision_dep.enabled and config.model.vision_dep.bubble
    return config


def dep_bubble_off() -> Trainer.Config:
    """DEP on, every encode inline in the tower stage's forward."""
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    config.model.vision_dep.bubble = False
    return config


def dep_prefetch() -> Trainer.Config:
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    config.model.vision_dep.bubble = False
    config.model.vision_dep.prefetch = 1
    return config


def dep_off() -> Trainer.Config:
    """No DEP: the default split, the tower in the first stage with the first layers."""
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    config.model.vision_dep.enabled = False
    config.model.vision_dep.bubble = False
    return config


def dep_mixed_pp2() -> Trainer.Config:
    """dp_shard 2 x pp2 (four stages, the tower alone on the first), even DP ranks text only."""
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 2
    set_rank_conditional_image_presence(config)
    return config
