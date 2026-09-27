"""Local recipe (never committed): the vit_dep cell at dp_shard 2 with even DP ranks text-only."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp4_vp2_vit_dep
from torchtitan_recipes.tests.multimodal import set_rank_conditional_image_presence


def dep_dp2_mixed() -> Trainer.Config:
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    config.parallelism.data_parallel_shard_degree = 2
    set_rank_conditional_image_presence(config)
    return config
