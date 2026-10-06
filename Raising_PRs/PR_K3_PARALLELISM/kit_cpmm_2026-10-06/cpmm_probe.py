"""Local cells for #4380 (dynamic CP of the Kimi K3 vision tower) and its main baseline; logbook kit only.

cell(): the Kimi K3 debug recipe (main's kimi_k3_debugmodel) as #4639's h100 cells build it. Env:
  CP_MODE      none | allgather (head-tail balancer) | ulysses   (default allgather)
  CP_DEGREE    context parallel degree (2); CP_DP dp_shard (1); CP_TP tensor parallel (1, sets EP = TP); AdamW everywhere
  CP_MINP      dynamic_cp_min_patches when the tree has it (256 = default; 128 splits cc12m-test's 192-patch images)
  CP_TYPECHECK 0/1 (1, as #4639's cells); CP_STEPS (100); seed 42, deterministic.
"""

import os

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan_recipes.tests import _set_spmd_typechecking


def cell():
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        KVAllGatherCPFlexInnerAttention,
        UlyssesCPFlexInnerAttention,
    )
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.kda import InnerKDA

    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    e = os.environ
    mode = e.get("CP_MODE", "allgather")
    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=e.get("CP_TYPECHECK", "1") == "1")
    config.parallelism.data_parallel_shard_degree = int(e.get("CP_DP", "1"))
    # DistMuon's recipe layouts name dp_shard, which CP turns into dp_shard_cp; every cell trains with AdamW.
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    tp = int(e.get("CP_TP", "1"))
    if tp > 1:
        config.parallelism.tensor_parallel_degree = tp
        config.parallelism.expert_parallel_degree = tp
    vision = config.model.vision_encoder
    if vision is not None and hasattr(vision, "dynamic_cp_min_patches"):
        vision.dynamic_cp_min_patches = int(e.get("CP_MINP", "256"))
    if mode != "none":
        config.parallelism.context_parallel_degree = int(e.get("CP_DEGREE", "2"))
        if mode == "allgather":
            config.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
            inner = KVAllGatherCPFlexInnerAttention
        else:
            config.parallelism.context_parallel_load_balancer = None
            inner = UlyssesCPFlexInnerAttention
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
    config.training.steps = int(e.get("CP_STEPS", "100"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
