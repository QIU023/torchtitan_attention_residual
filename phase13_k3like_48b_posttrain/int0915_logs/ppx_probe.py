"""Scratch recipes for the PP offload and PP balance stacks on the 4312 head (pp2 x vp2, the b200 recipe)."""
import dataclasses
from functools import partial

from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp2_vp2


def _with_pipelining(config: Trainer.Config, **kwargs) -> Trainer.Config:
    from torchtitan.models.kimi_k3.parallelize import pipeline_kimi_k3

    assert config.model_spec is not None
    config.model_spec = dataclasses.replace(
        config.model_spec, pipelining_fn=partial(pipeline_kimi_k3, **kwargs)
    )
    return config


def pp2_vp2_plain() -> Trainer.Config:
    return kimi_k3_debugmodel_pp2_vp2()


def pp2_vp2_offload() -> Trainer.Config:
    return _with_pipelining(kimi_k3_debugmodel_pp2_vp2(), attn_res_cache_offload=True)


def pp2_vp2_balance() -> Trainer.Config:
    from torchtitan.models.kimi_k3.pp_balance import PPBalanceKnobs

    return _with_pipelining(
        kimi_k3_debugmodel_pp2_vp2(),
        pp_balance=PPBalanceKnobs(pp_balance_source_ranks=(0,), pp_balance_dest_rank=1),
    )
