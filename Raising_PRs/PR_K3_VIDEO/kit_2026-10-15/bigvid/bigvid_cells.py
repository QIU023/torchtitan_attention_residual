"""Video at the released per-frame cap through the stack's debug model with DEP and dynamic CP (local kit, never committed).

Runs on a scratch worktree of the stack plus the video commit. synthetic-video-text frames are stretched to K3V_SIDE
(1792, the released 16384-patch-per-frame cap) and split into chunks of max_num_frames; a 4-frame chunk is 65536
patches and 4096 tokens. K3V_CELL degrees as in big_cells (pp > 1 runs Interleaved1F1B, four stages per rank);
K3V_DEP=1, K3V_DYNCP=<n>; seq 8704, 4 pipeline micro-batches of one sample each per dp rank, AdamW, seed 42.
"""

import os
from dataclasses import replace
from functools import partial

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import ParallelismConfig


def _degrees(spec: str) -> dict[str, int]:
    degrees = {"dp": 1, "ep": 1, "pp": 1, "cp": 1}
    for item in filter(None, spec.split(",")):
        key, value = item.split("=")
        degrees[key] = int(value)
    return degrees


def _big_video_sample(sample, tokenizer, *, video_root, side, frames_per_chunk, **kwargs):
    import torch
    import torch.nn.functional as F
    from torchtitan.components.data.multimodal.mm_datasets import _process_chunked_video_text_sample
    from torchtitan.components.data.multimodal.utils.video import load_npy_video_frames

    frames = load_npy_video_frames(video_root / sample["video_path"])
    frames = F.interpolate(frames.permute(0, 3, 1, 2).float(), size=(side, side), mode="bicubic", align_corners=False)
    frames = frames.clamp(0, 255).round().to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).contiguous()
    return _process_chunked_video_text_sample(
        text=sample["text"], frames=frames, tokenizer=tokenizer, frames_per_chunk=frames_per_chunk, fps=8.0, **kwargs
    )


def _build(degrees: dict[str, int]):
    from torchtitan_recipes.tests.datasets import synthetic_multimodal
    from torchtitan_recipes.tests.models.kimi_k3 import _kimi_k3_multimodal_dataloader, kimi_k3_debugmodel

    seq = int(os.environ.get("K3V_SEQ", "8704"))
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
    config.training.steps = int(os.environ.get("K3V_STEPS", "8"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.optim.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    dataset = synthetic_multimodal.SYNTHETIC_MM_DATASETS["synthetic-video-text"]
    sample_processor = partial(
        _big_video_sample,
        video_root=synthetic_multimodal._VIDEO_ROOT,
        side=int(os.environ.get("K3V_SIDE", "1792")),
        frames_per_chunk=config.model.vision_encoder.max_num_frames,
    )
    loader = _kimi_k3_multimodal_dataloader(
        replace(dataset, processor=replace(dataset.processor, sample_processor=sample_processor))
    )
    loader = replace(
        loader,
        dataset=replace(
            loader.dataset,
            processor=replace(loader.dataset.processor, max_patches=16384, max_patches_per_side=512),
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
    if os.environ.get("K3V_DEP") == "1":
        config.model.vision_dep.enabled = True
    dyncp = int(os.environ.get("K3V_DYNCP", "0"))
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
