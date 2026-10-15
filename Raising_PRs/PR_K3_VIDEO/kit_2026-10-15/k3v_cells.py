"""K3 video PR matrix cells on latest main (local kit, never committed).

K3V_CELL: degrees "dp=2,tp=2,ep=2,pp=2,cp=1,vpp=1,tc=0" (missing keys are 1, tc 0). SP is on under TP unless CP is on.
K3V_SEED: seed checkpoint folder (model only). K3V_STEPS: steps (10). K3V_ATTN: flex (default) or varlen.
Kimi K3 debug model, synthetic-video-text in 4-frame chunks, seq 256 (512 at CP4), 4096 tokens per step in
one-sequence micro-batches per dp rank, 8 pipeline micro-batches, AdamW, seed 42, deterministic.
"""

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import ParallelismConfig


def _degrees(spec: str) -> dict[str, int]:
    degrees = {"dp": 1, "tp": 1, "ep": 1, "pp": 1, "cp": 1, "vpp": 1, "tc": 0, "mix": 0}
    for item in filter(None, spec.split(",")):
        key, value = item.split("=")
        degrees[key] = int(value)
    return degrees


def _build(degrees: dict[str, int]):
    from torchtitan_recipes.tests import _set_spmd_typechecking
    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel_video

    cp, tp = degrees["cp"], degrees["tp"]
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=degrees["dp"],
        expert_parallel_degree=degrees["ep"],
        pipeline_parallel_degree=degrees["pp"],
        enable_sequence_parallel=tp > 1 and cp == 1,
    )
    seq_len = max(256, 128 * cp)
    config = kimi_k3_debugmodel_video(
        seq_len=seq_len, parallelism=parallelism, attn_backend=os.environ.get("K3V_ATTN", "flex")
    )
    config.parallelism.tensor_parallel_degree = tp
    config.parallelism.context_parallel_degree = cp
    if degrees["pp"] > 1:
        config.parallelism.num_pp_microbatches = 8
    if degrees["vpp"] == 2:
        config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    else:
        assert degrees["vpp"] == 1, degrees
    config.training.num_tokens_per_train_step = 4096
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    config.training.steps = int(os.environ.get("K3V_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    config.optim.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    if degrees["tc"]:
        _set_spmd_typechecking(config, typechecking=True)
    if degrees["mix"]:
        # cc12m-test images and the chunked synthetic videos, one sample per micro-batch, equal weights.
        from dataclasses import replace

        from torchtitan.components.data.dataset import DatasetMixConfig, WeightedDataset
        from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
        from torchtitan_recipes.tests.models.kimi_k3 import _kimi_k3_multimodal_dataloader

        images = _kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]).dataset
        config.dataloader = replace(
            config.dataloader,
            dataset=DatasetMixConfig(
                datasets=(WeightedDataset(dataset=images), WeightedDataset(dataset=config.dataloader.dataset))
            ),
        )
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
    config = _build(_degrees(os.environ["K3V_CELL"]))
    config.checkpointer = CheckpointManager.Config(
        interval=100000, initial_load_path=os.environ["K3V_SEED"], initial_load_model_only=True
    )
    return config


def seed():
    config = _build(_degrees(""))
    config.checkpointer = CheckpointManager.Config(folder=os.environ["K3V_SEED_OUT"])
    config.create_seed_checkpoint = True
    config.training.steps = 1
    return config


def smoke():
    config = _build(_degrees(os.environ.get("K3V_CELL", "")))
    config.checkpointer = None
    return config
