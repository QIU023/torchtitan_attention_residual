"""Local DEP recipes (logbook kit only, never committed) for the DEP tree on upstream main after the Python-config
loader (dep_review1 a93cd48ea on db050eb3f): the port of kit_overnight_2026-09-29/dep_ratio/dep_ratio_local.py.

Same cells as that file: the B200 vision_dep cell turned into pp4 x vpp4 x tp1 x ep1 with core's 16 stage split and the
model's end modules pinned, the debug text widened to DEPW_DIM (17 layers), the tower from DEPV_TOWER (base: the debug
tower at its own width; k3: the released shape), the t2i1024_k4 webdataset with fixed-square images. Knobs as before:
DEPR_DATA, DEPR_RES, DEPR_SEQ, DEPR_NMAX, DEPR_IMG_PER16, DEPR_AC=full, DEPV_COST_RATIO.

The loader takes no field flags any more, so what the old scripts passed on the command line is read here:
DEPN_STEPS (steps), DEPN_MBS (pipeline micro-batches; tokens per step = DEPN_MBS x DEPR_SEQ), DEPN_DET=1 (seed 42,
deterministic), DEPN_PROFILE=1 (one profiled step at step 10, traces under <output-dir>/traces), DEPN_STATIC=1
(torch._dynamo automatic_dynamic_shapes off, for DEP on/off numerics: the tower and the text share compiled flex
attention), DEPN_MEM_OUT (per-step peaks per rank through kit_h100_2026-09-29/dep/dep4.py).
"""

import os

import torch

from torchtitan.trainer import Trainer

if os.environ.get("DEPN_STATIC") == "1":
    torch._dynamo.config.automatic_dynamic_shapes = False

if os.environ.get("DEPN_MEM_OUT"):
    import dep4  # noqa: F401

PATCH, MERGE = 14, 2
_SIDES = {224: 224, 448: 448, 768: 756, 1024: 1008}


def _side() -> int:
    return _SIDES[int(os.environ.get("DEPR_RES", "1024"))]


def _seq() -> int:
    return int(os.environ.get("DEPR_SEQ", "2048"))


def image_tokens(side: int) -> int:
    return (side // PATCH // MERGE) ** 2


def fixed_square_resize(height, width, *, patch_size, merge_size, **_):
    side = _side()
    assert side % (patch_size * merge_size) == 0
    return side, side, 0, 0


def _images_cap() -> int:
    fit = max(0, (_seq() - 96) // (image_tokens(_side()) + 2))
    cap = os.environ.get("DEPR_NMAX")
    return min(fit, int(cap)) if cap else fit


def _carries_images(key: str) -> bool | None:
    per16 = os.environ.get("DEPR_IMG_PER16")
    if per16 is None:
        return None
    i, k = int(key.lstrip("s")), int(per16)
    return (i + 1) * k // 16 > i * k // 16


def process_ratio_sample(sample, **kwargs):
    from torchtitan.hf_datasets.multimodal.mm_datasets import _process_mm_sample

    meta = sample.get("json") or {}
    carries = _carries_images(meta.get("key", ""))
    stored = int(meta.get("n_images", 0)) if carries is None else (4 if carries else 0)
    n = min(stored, _images_cap())
    images = [sample[f"img{j}.jpg"] for j in range(n)]
    texts = [None] * n + [sample.get("txt", "")]
    return _process_mm_sample(texts=texts, images=images + [None], **kwargs)


def _not_none(sample):
    return sample is not None


def _dataloader():
    from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
    from torchtitan.components.data.sources import HuggingFaceStreamingSource
    from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
    from torchtitan.hf_datasets.multimodal.mm_datasets import MultiModalProcessor

    side = _side()
    per_side = side // PATCH
    processor = MultiModalProcessor.Config(
        sample_processor=process_ratio_sample,
        patch_size=PATCH,
        temporal_patch_size=1,
        spatial_merge_size=MERGE,
        resize_fn=fixed_square_resize,
        max_patches=max(5184, per_side * per_side),
        max_patches_per_side=max(72, per_side),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    dataset = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path=os.environ.get("DEPR_DATA", "/workspace/dep_data/t2i1024_k4"),
            split="train",
            load_dataset_kwargs={"data_files": {"train": "*.tar"}},
        ),
        processor=processor,
        post_filters=(_not_none,),
    )
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=MultiModalCollator.Config(
            patch_size=PATCH,
            temporal_patch_size=1,
            spatial_merge_size=MERGE,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )


def _tower(dim: int, s: int):
    from torchtitan.models.kimi_k3.flavors import _vision_encoder_config

    tower = os.environ.get("DEPV_TOWER", "debug")
    if tower == "k3":
        return _vision_encoder_config(
            text_dim=dim, dim=1024, qkv_dim=1536, hidden_dim=4096, num_layers=27, num_heads=12,
            init_pos_emb_height=64, init_pos_emb_width=64)
    if tower == "base":
        return _vision_encoder_config(
            text_dim=dim, dim=256, qkv_dim=512, hidden_dim=512, num_layers=2, num_heads=4,
            init_pos_emb_height=32, init_pos_emb_width=32)
    return _vision_encoder_config(
        text_dim=dim, dim=256 * s, qkv_dim=512 * s, hidden_dim=512 * s, num_layers=2, num_heads=4 * s,
        init_pos_emb_height=32, init_pos_emb_width=32)


def _widened(seq_len: int):
    from torchtitan.models.kimi_k3.flavors import _kimi_k3_config

    dim = int(os.environ["DEPW_DIM"])
    s = dim // 256
    assert dim == 256 * s
    return _kimi_k3_config(
        max_context_length=seq_len, dim=dim, vocab_size=2048, num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16}, attn_res_block_size=4, num_heads=4 * s, q_lora_rank=128 * s,
        kv_lora_rank=64 * s, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64, kda_head_dim=128,
        conv_kernel_size=4, dense_hidden_dim=512 * s, latent_dim=128 * s, expert_hidden_dim=128 * s, num_experts=8,
        top_k=2, num_shared_experts=2, vision_encoder=_tower(dim, s), attn_backend="flex")


def _pp4_vpp4(config: Trainer.Config) -> None:
    from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part
    from torchtitan.models.kimi_k3.model import KimiK3Model

    p = config.parallelism
    p.tensor_parallel_degree = 1
    p.enable_sequence_parallel = False
    p.expert_parallel_degree = 1
    p.pipeline_parallel_degree = 4
    split = _generate_llm_fqn_per_model_part(
        4 * p.pipeline_parallel_degree, len(config.model.layers),
        p.pipeline_parallel_first_stage_less_layers, p.pipeline_parallel_last_stage_less_layers)
    split[0][:0] = KimiK3Model.pipeline_first_stage_module_fqns
    split[-1].extend(KimiK3Model.pipeline_last_stage_module_fqns)
    p.pipeline_parallel_module_fqns_per_model_part = split


def _base(widened: bool) -> Trainer.Config:
    from torchtitan_recipes.tests.suites.b200 import kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4

    config = kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4()
    seq = _seq()
    if widened:
        config.model = _widened(seq)
    config.parallelism.data_parallel_shard_degree = 1
    if os.environ.get("DEPR_LAYOUT", "pp4vpp4") == "pp4vpp4":
        _pp4_vpp4(config)
    config.training.max_context_length = seq
    config.training.num_tokens_per_microbatch_per_dp_rank = seq
    config.model.max_context_length = max(config.model.max_context_length, seq)
    config.dataloader = _dataloader()
    if os.environ.get("DEPR_AC") == "full":
        from torchtitan.distributed.activation_checkpoint import FullAC

        config.activation_checkpoint = FullAC.Config()
    return config


def _run(config: Trainer.Config) -> Trainer.Config:
    config.training.steps = int(os.environ.get("DEPN_STEPS", "2"))
    config.metrics.log_freq = 1
    mbs = os.environ.get("DEPN_MBS")
    if mbs:
        config.parallelism.num_pp_microbatches = int(mbs)
        config.training.num_tokens_per_train_step = int(mbs) * _seq()
    if os.environ.get("DEPN_DET") == "1":
        config.debug.seed = 42
        config.debug.deterministic = True
    if os.environ.get("DEPN_PROFILE") == "1":
        config.profiler.enable_profiling = True
        config.profiler.profile_freq = 10
        config.profiler.profiler_warmup = 2
        config.profiler.profiler_active = 1
        config.profiler.save_traces_folder = "traces"
    return config


def _mode(config: Trainer.Config, enabled: bool, bubble: bool) -> Trainer.Config:
    config.model.vision_dep.enabled = enabled
    config.model.vision_dep.bubble = bubble
    if os.environ.get("DEPV_COST_RATIO"):
        config.model.vision_dep.bubble_cost_ratio = float(os.environ["DEPV_COST_RATIO"])
    return _run(config)


def w_dep_off() -> Trainer.Config:
    return _mode(_base(True), False, False)


def w_dep_k25() -> Trainer.Config:
    return _mode(_base(True), True, False)


def w_dep_bubble() -> Trainer.Config:
    return _mode(_base(True), True, True)
