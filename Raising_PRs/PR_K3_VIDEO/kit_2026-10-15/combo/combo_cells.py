"""DEP and dynamic CP together on the stack (local kit, never committed).

K3C_CELL: degrees "dp=2,ep=2,pp=2,cp=2" (pp > 1 runs Interleaved1F1B with four stages per rank, core's split).
K3C_DEP=1 turns on vision_dep; K3C_DYNCP=<n> sets dynamic_cp_min_patches; K3C_MAXP=<n> raises the image patch budget
(cc12m-test native size, at most 64 patches a side). Debug model, seq 2048 (one sample per micro-batch),
4 pipeline micro-batches, AdamW, seed 42, deterministic; K3C_SEED loads a model-only seed, seed() writes it.
"""

import os
from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import ParallelismConfig


def _degrees(spec: str) -> dict[str, int]:
    degrees = {"dp": 1, "ep": 1, "pp": 1, "cp": 1}
    for item in filter(None, spec.split(",")):
        key, value = item.split("=")
        degrees[key] = int(value)
    return degrees


def _build(degrees: dict[str, int]):
    from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
    from torchtitan_recipes.tests.models.kimi_k3 import _kimi_k3_multimodal_dataloader, kimi_k3_debugmodel

    cp, pp = degrees["cp"], degrees["pp"]
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=degrees["dp"],
        expert_parallel_degree=degrees["ep"],
        pipeline_parallel_degree=pp,
    )
    config = kimi_k3_debugmodel(seq_len=2048, parallelism=parallelism)
    config.parallelism.context_parallel_degree = cp
    if pp > 1:
        from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part
        from torchtitan.models.kimi_k3.model import KimiK3Model

        config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
        config.parallelism.num_pp_microbatches = 4
        split = _generate_llm_fqn_per_model_part(
            4 * pp,
            len(config.model.layers),
            config.parallelism.pipeline_parallel_first_stage_less_layers,
            config.parallelism.pipeline_parallel_last_stage_less_layers,
        )
        split[0][:0] = KimiK3Model.pipeline_first_stage_module_fqns
        split[-1].extend(KimiK3Model.pipeline_last_stage_module_fqns)
        config.parallelism.pipeline_parallel_module_fqns_per_model_part = split
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 2048 * 4 * degrees["dp"]
    config.training.steps = int(os.environ.get("K3C_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    config.optim.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    maxp = int(os.environ.get("K3C_MAXP", "0"))
    if maxp:
        loader = _kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"])
        dataset = loader.dataset
        loader = replace(
            loader,
            dataset=replace(dataset, processor=replace(dataset.processor, max_patches=maxp, max_patches_per_side=64)),
        )
        config.dataloader = loader
    if os.environ.get("K3C_DEP") == "1":
        config.model.vision_dep.enabled = True
    dyncp = int(os.environ.get("K3C_DYNCP", "0"))
    if dyncp:
        config.model.vision_encoder.dynamic_cp_min_patches = dyncp
    if cp > 1:
        from torchtitan.config.transform import apply_transforms, ContextParallelTransform
        from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
        from torchtitan.models.common.attention import FlexInnerAttention
        from torchtitan.models.common.attention.cp_attention import KVAllGatherCPFlexInnerAttention
        from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
        from torchtitan.models.common.attention.kda import InnerKDA

        config.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
        config = apply_transforms(
            config,
            [
                ContextParallelTransform(
                    inner_attention_map={
                        FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                        InnerKDA: ContextParallelInnerKDA,
                    }
                )
            ],
        )
    return config


def cell():
    config = _build(_degrees(os.environ["K3C_CELL"]))
    config.checkpointer = CheckpointManager.Config(
        interval=100000, initial_load_path=os.environ["K3C_SEED"], initial_load_model_only=True
    )
    return config


def seed():
    config = _build(_degrees(""))
    config.checkpointer = CheckpointManager.Config(folder=os.environ["K3C_SEED_OUT"])
    config.create_seed_checkpoint = True
    config.training.steps = 1
    return config
