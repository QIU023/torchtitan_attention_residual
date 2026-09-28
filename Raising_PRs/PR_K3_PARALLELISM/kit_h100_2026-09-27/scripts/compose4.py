"""Local recipe (never committed): the B200 composition cell without FSDP, tp2 x ep2 x pp2 x vpp4 on 4 GPUs."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4


def compose4() -> Trainer.Config:
    config = kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4()
    config.parallelism.data_parallel_shard_degree = 1
    return config
