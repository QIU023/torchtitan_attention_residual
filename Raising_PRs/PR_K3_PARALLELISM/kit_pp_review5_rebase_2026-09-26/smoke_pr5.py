"""Local smoke recipe (never committed): pp4 x vp4 on the Kimi K3 debug model.

The composition cell's recipe at pp4 without FSDP/TP/EP: 16 stages from core's
_generate_llm_fqn_per_model_part with the model's end modules pinned, AdamW.
"""

from torchtitan.components.optimizer import default_adamw
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _set_spmd_typechecking


def kimi_k3_pp4_vpp4() -> Trainer.Config:
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
    )
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
    from torchtitan.models.kimi_k3.model import KimiK3Model

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=False)
    parallelism = config.parallelism
    parallelism.pipeline_parallel_degree = 4
    parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    parallelism.num_pp_microbatches = 4
    split = _generate_llm_fqn_per_model_part(
        4 * parallelism.pipeline_parallel_degree,
        len(config.model.layers),
        parallelism.pipeline_parallel_first_stage_less_layers,
        parallelism.pipeline_parallel_last_stage_less_layers,
    )
    split[0][:0] = KimiK3Model.pipeline_first_stage_module_fqns
    split[-1].extend(KimiK3Model.pipeline_last_stage_module_fqns)
    parallelism.pipeline_parallel_module_fqns_per_model_part = split
    print(f"SMOKE_SPLIT {len(split)} stages: {split}", flush=True)
    config.optimizer = default_adamw(lr=8e-4)
    return config


def kimi_k3_pp4_vpp4_default_optimizer() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_pp4_vpp4()
    config.optimizer = kimi_k3_debugmodel().optimizer
    return config


def kimi_k3_pp4_vpp4_default_optimizer_layer_per_stage() -> Trainer.Config:
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
    )
    from torchtitan.models.kimi_k3.model import KimiK3Model

    config = kimi_k3_pp4_vpp4_default_optimizer()
    parallelism = config.parallelism
    parallelism.pipeline_parallel_last_stage_less_layers = 0
    split = _generate_llm_fqn_per_model_part(
        4 * parallelism.pipeline_parallel_degree,
        len(config.model.layers),
        parallelism.pipeline_parallel_first_stage_less_layers,
        parallelism.pipeline_parallel_last_stage_less_layers,
    )
    split[0][:0] = KimiK3Model.pipeline_first_stage_module_fqns
    split[-1].extend(KimiK3Model.pipeline_last_stage_module_fqns)
    parallelism.pipeline_parallel_module_fqns_per_model_part = split
    print(f"SMOKE_SPLIT2 {len(split)} stages: {split}", flush=True)
    return config
