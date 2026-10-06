"""Integration-tree matrix cells (main + Elfie + the PR lines + #4380), logbook kit only, never committed.

cell(): the Kimi K3 debug recipe with AdamW (lr 8e-4), seed 42, deterministic. Env (defaults in brackets):
  INT_FLAVOR debugmodel | rl | report_arch | k3mini [debugmodel]; INT_QAT 1 = Elfie's MX QAT recipe [0]
  INT_DP dp_shard [1]; INT_TP [1] (EP raised to TP); INT_EP [1]; INT_SP [1]
  INT_CP degree [1] with INT_CPMODE allgather | ulysses [allgather]; INT_MINP dynamic_cp_min_patches [256]
  INT_PP [1], INT_VPP stages per rank [2], INT_MB microbatches [4], INT_SCHED [Interleaved1F1B]
  INT_DEP 1 = vision_dep [0], INT_DEP_BUBBLE [0]; INT_PPMEM none | all | planned [none], INT_BALANCE [0]
  INT_TYPECHECK [0]; INT_STEPS [10]
"""

import os

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan_recipes.tests import _set_spmd_typechecking


def cell():
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part
    from torchtitan.models.kimi_k3 import build_model_config
    from torchtitan.models.kimi_k3.model import KimiK3Model

    from torchtitan_recipes.tests.models import kimi_k3 as recipes

    e = os.environ.get
    if e("INT_QAT", "0") == "1":
        config = recipes.kimi_k3_debugmodel_mx_qat()
        config.checkpointer = None
    else:
        config = recipes.kimi_k3_debugmodel()
    flavor = e("INT_FLAVOR", "debugmodel")
    if flavor != "debugmodel":
        config.model = build_model_config(flavor, seq_len=config.training.max_context_length)
        config.loss = ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(config.model))
        )
    _set_spmd_typechecking(config, typechecking=e("INT_TYPECHECK", "0") == "1")
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    p = config.parallelism
    p.data_parallel_shard_degree = int(e("INT_DP", "1"))
    tp = int(e("INT_TP", "1"))
    p.tensor_parallel_degree = tp
    p.enable_sequence_parallel = e("INT_SP", "1") == "1"
    p.expert_parallel_degree = max(int(e("INT_EP", "1")), tp)
    vision = config.model.vision_encoder
    if vision is not None and hasattr(vision, "dynamic_cp_min_patches"):
        vision.dynamic_cp_min_patches = int(e("INT_MINP", "256"))
    cp = int(e("INT_CP", "1"))
    if cp > 1:
        from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
        from torchtitan.models.common.attention import FlexInnerAttention
        from torchtitan.models.common.attention.cp_attention import (
            KVAllGatherCPFlexInnerAttention,
            UlyssesCPFlexInnerAttention,
        )
        from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
        from torchtitan.models.common.attention.kda import InnerKDA

        p.context_parallel_degree = cp
        allgather = e("INT_CPMODE", "allgather") == "allgather"
        p.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config() if allgather else None
        inner = KVAllGatherCPFlexInnerAttention if allgather else UlyssesCPFlexInnerAttention
        config = apply_transforms(
            config,
            [
                ContextParallelTransform(
                    inner_attention_map={FlexInnerAttention: inner, InnerKDA: ContextParallelInnerKDA}
                )
            ],
        )
        p = config.parallelism
    pp = int(e("INT_PP", "1"))
    if pp > 1:
        p.pipeline_parallel_degree = pp
        p.pipeline_parallel_schedule = e("INT_SCHED", "Interleaved1F1B")
        p.num_pp_microbatches = int(e("INT_MB", "4"))
        stages = int(e("INT_VPP", "2")) * pp
        split = _generate_llm_fqn_per_model_part(
            stages,
            len(config.model.layers),
            p.pipeline_parallel_first_stage_less_layers,
            p.pipeline_parallel_last_stage_less_layers,
        )
        split[0][:0] = KimiK3Model.pipeline_first_stage_module_fqns
        split[-1].extend(KimiK3Model.pipeline_last_stage_module_fqns)
        p.pipeline_parallel_module_fqns_per_model_part = split
        config.model.vision_dep.enabled = e("INT_DEP", "0") == "1"
        config.model.vision_dep.bubble = e("INT_DEP_BUBBLE", "0") == "1"
        config.model.pp_memory.cpu_offload = e("INT_PPMEM", "none")
        config.model.pp_memory.balance = e("INT_BALANCE", "0") == "1"
    config.training.steps = int(e("INT_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
