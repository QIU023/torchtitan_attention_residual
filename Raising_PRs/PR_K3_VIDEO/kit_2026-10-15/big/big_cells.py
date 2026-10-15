"""3584 x 3584 images through the stack's debug model with DEP and dynamic CP (local kit, never committed).

K3B_CELL: degrees "pp=2,cp=4,ep=4" (dp, ep, pp, cp; pp > 1 runs Interleaved1F1B with four stages per rank, core's split).
K3B_DEP=1 turns on vision_dep; K3B_DYNCP=<n> sets dynamic_cp_min_patches. Debug model unchanged except its flavor cap
(16384) is raised so seq K3B_SEQ (16896) holds one 16384-token image plus its caption; data is cc12m-test stretched to
3584 x 3584 (big/data), processed with the released caps (65536 patches, 512 a side); 4 pipeline micro-batches of one
sample each per dp rank, AdamW, seed 42. K3B_SEED loads a model-only seed; seed() writes it.
"""

import os
from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import ParallelismConfig

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _degrees(spec: str) -> dict[str, int]:
    degrees = {"dp": 1, "ep": 1, "pp": 1, "cp": 1}
    for item in filter(None, spec.split(",")):
        key, value = item.split("=")
        degrees[key] = int(value)
    return degrees


def _build(degrees: dict[str, int]):
    from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
    from torchtitan.models.kimi_k3 import flavors
    from torchtitan_recipes.tests.models.kimi_k3 import _kimi_k3_multimodal_dataloader, kimi_k3_debugmodel

    flavors.MODEL_FLAVORS["debugmodel"] = (flavors._debugmodel, 32768)
    seq = int(os.environ.get("K3B_SEQ", "16896"))
    cp, pp = degrees["cp"], degrees["pp"]
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=degrees["dp"],
        expert_parallel_degree=degrees["ep"],
        pipeline_parallel_degree=pp,
    )
    config = kimi_k3_debugmodel(seq_len=seq, parallelism=parallelism)
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
    config.training.num_tokens_per_microbatch_per_dp_rank = seq
    config.training.num_tokens_per_train_step = seq * 4 * degrees["dp"]
    config.training.steps = int(os.environ.get("K3B_STEPS", "8"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.optim.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    base = MM_DATASETS["cc12m-test"]
    big = replace(base, source=replace(base.source, path=_DATA))
    loader = _kimi_k3_multimodal_dataloader(big)
    loader = replace(
        loader,
        dataset=replace(
            loader.dataset,
            processor=replace(loader.dataset.processor, max_patches=65536, max_patches_per_side=512),
        ),
    )
    # titan's defaults (16 read threads, 500 prefetched and 1000 shuffle-buffered samples) hold processed float32
    # samples per rank; at 3584 x 3584 (about 300 MB each) they fill the 241 GiB cgroup within minutes.
    import grain.python as grain

    loader = replace(
        loader,
        read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=2),
        streaming_shuffle_buffer_size=4,
        num_prefetch_microbatches=1,
    )
    config.dataloader = loader
    if os.environ.get("K3B_DEP") == "1":
        config.model.vision_dep.enabled = True
    dyncp = int(os.environ.get("K3B_DYNCP", "0"))
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
    config = _build(_degrees(os.environ["K3B_CELL"]))
    config.checkpointer = CheckpointManager.Config(
        interval=100000, initial_load_path=os.environ["K3B_SEED"], initial_load_model_only=True
    )
    return config


def seed():
    config = _build(_degrees(""))
    config.checkpointer = CheckpointManager.Config(folder=os.environ["K3B_SEED_OUT"])
    config.create_seed_checkpoint = True
    config.training.steps = 1
    return config
