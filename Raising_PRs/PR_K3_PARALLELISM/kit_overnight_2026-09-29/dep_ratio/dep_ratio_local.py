"""Local recipes (never committed) for the DEP ratio study on the DEP tree a03f74981.

Data: the local webdataset built by build_dataset.py (DEPR_DATA, default /workspace/dep_data/t2i1024_k4): every
sample stores 4 images of 1024 x 1024 and uses the first n of them (n from its json, 0 to 4), capped so the
sample fits in DEPR_SEQ tokens. Every image is resized to a fixed square of DEPR_RES (224 -> 224, 448 -> 448, 768 -> 756,
1024 -> 1008 px, sides a multiple of patch 14 x merge 2), upsampling allowed (the K3 recipe's
resize_to_navit_patch_grid only shrinks and caps an image at 256 patches).

Layout: the B200 vision_dep cell without FSDP (pp2 x vpp4 x tp2 x ep2), as kit_h100_2026-09-29/dep/dep4.py.
dep_off / dep_k25 / dep_bubble at the debug width; w_* widen the debug model to DEPW_DIM.
Knobs: DEPR_DATA, DEPR_RES, DEPR_SEQ (tokens per micro-batch, default 2048), DEPR_NMAX (images per sample cap).
"""

import os

from torchtitan.trainer import Trainer

if os.environ.get("DEPN_MEM_OUT"):
    import dep4  # noqa: F401  # per-step peaks per rank; kit_h100_2026-09-29/dep on PYTHONPATH

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


def process_ratio_sample(sample, **kwargs):
    from torchtitan.hf_datasets.multimodal.mm_datasets import _process_mm_sample

    meta = sample.get("json") or {}
    n = min(int(meta.get("n_images", 0)), _images_cap())
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


def _widened(attn_backend="flex", converters=None, moe_comm_backend="standard", *, enable_sp, seq_len=None):
    from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config

    dim = int(os.environ["DEPW_DIM"])
    s = dim // 256
    assert dim == 256 * s and converters is None
    return _kimi_k3_config(
        max_context_length=seq_len or 2048, dim=dim, enable_sp=enable_sp, moe_comm_backend=moe_comm_backend,
        vocab_size=2048, num_layers=17, full_attention_layers={3, 7, 11, 15, 16}, attn_res_block_size=4,
        num_heads=4 * s, q_lora_rank=128 * s, kv_lora_rank=64 * s, qk_nope_head_dim=64, qk_rope_head_dim=32,
        v_head_dim=64, kda_head_dim=128, conv_kernel_size=4, dense_hidden_dim=512 * s, latent_dim=128 * s,
        expert_hidden_dim=128 * s, num_experts=8, top_k=2, num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim, dim=256 * s, qkv_dim=512 * s, hidden_dim=512 * s, num_layers=2, num_heads=4 * s,
            init_pos_emb_height=32, init_pos_emb_width=32),
        attn_backend=attn_backend)


def _base(widened: bool = False) -> Trainer.Config:
    from torchtitan.models.kimi_k3 import config_registry
    from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4

    original = config_registry.model_registry
    if widened:
        config_registry.model_registry = lambda flavor, **kw: _widened(**kw)
    try:
        config = kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4()
    finally:
        config_registry.model_registry = original
    config.parallelism.data_parallel_shard_degree = 1
    seq = _seq()
    config.training.max_context_length = seq
    config.training.num_tokens_per_microbatch_per_dp_rank = seq
    config.model.max_context_length = max(config.model.max_context_length, seq)
    config.dataloader = _dataloader()
    return config


def _mode(config: Trainer.Config, enabled: bool, bubble: bool) -> Trainer.Config:
    config.model.vision_dep.enabled = enabled
    config.model.vision_dep.bubble = bubble
    return config


def dep_off() -> Trainer.Config:
    return _mode(_base(), False, False)


def dep_k25() -> Trainer.Config:
    return _mode(_base(), True, False)


def dep_bubble() -> Trainer.Config:
    return _mode(_base(), True, True)


def w_dep_off() -> Trainer.Config:
    return _mode(_base(True), False, False)


def w_dep_k25() -> Trainer.Config:
    return _mode(_base(True), True, False)


def w_dep_bubble() -> Trainer.Config:
    return _mode(_base(True), True, True)
