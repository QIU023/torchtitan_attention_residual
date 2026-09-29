"""Local recipes (never committed): PR 4312's H100 matrix flavors on main 5dc97a3e7. c4 docs as text-only
256-token rows, AdamW(lr=8e-4) as the matrix ran, every cell loading the seed checkpoint from
<dump_folder>/checkpoint and never saving; the 8- and 16-stage splits are the matrix's own."""

import os

from torchtitan.trainer import Trainer

_ROW_TOKENS = 256

_EIGHT_STAGES = [
    ["vision_encoder", "tok_embeddings", "layers.0", "layers.1"],
    ["layers.2", "layers.3", "layers.4"],
    ["layers.5", "layers.6", "layers.7"],
    ["layers.8", "layers.9"],
    ["layers.10", "layers.11"],
    ["layers.12", "layers.13"],
    ["layers.14", "layers.15"],
    ["layers.16", "norm", "lm_head", "output_res_proj", "output_res_norm"],
]

_SIXTEEN_STAGES = [
    ["vision_encoder", "tok_embeddings", "layers.0"],
    ["layers.1", "layers.2"],
    ["layers.3", "layers.4"],
    *[[f"layers.{i}"] for i in range(5, 17)],
    ["norm", "lm_head", "output_res_proj", "output_res_norm"],
]


def _process_c4_text_sample(sample, **kwargs):
    from torchtitan.hf_datasets.multimodal.mm_datasets import _process_mm_sample

    out = _process_mm_sample(texts=[sample["text"][:4000]], images=[None], **kwargs)
    if out is None:
        return None
    n = _ROW_TOKENS - 1
    for key in ("input_ids", "labels", "positions"):
        out[key] = out[key][:n]
    return out


def _not_none(sample):
    return sample is not None


def kimi_k3_debugmodel_c4() -> Trainer.Config:
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.data import SingleDatasetConfig
    from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
    from torchtitan.components.optimizer import AdamW, OptimizersContainer
    from torchtitan.hf_datasets.multimodal.mm_datasets import MultiModalProcessor
    from torchtitan.models.kimi_k3.config_registry import (
        _kimi_k3_multimodal_dataloader,
        kimi_k3_debugmodel,
    )
    from torchtitan_recipes.tests import _set_spmd_typechecking

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=False)
    config.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    config.checkpointer = CheckpointManager.Config(interval=100000)
    config.dataloader = _kimi_k3_multimodal_dataloader(
        SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                split="train",
                load_dataset_kwargs={"data_files": "tests/assets/c4_test/data.json"},
            ),
            processor=MultiModalProcessor.Config(sample_processor=_process_c4_text_sample),
            post_filters=(_not_none,),
        )
    )
    return config


def _naive(config: Trainer.Config) -> Trainer.Config:
    os.environ["ATTN_RES_NAIVE"] = "1"
    return config


def _stages(config: Trainer.Config, split) -> Trainer.Config:
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = [list(s) for s in split]
    return config


def kimi_k3_debugmodel_c4_pp_naive() -> Trainer.Config:
    return _naive(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_8stages() -> Trainer.Config:
    return _stages(kimi_k3_debugmodel_c4(), _EIGHT_STAGES)


def kimi_k3_debugmodel_c4_8stages_naive() -> Trainer.Config:
    return _naive(_stages(kimi_k3_debugmodel_c4(), _EIGHT_STAGES))


def kimi_k3_debugmodel_c4_16stages() -> Trainer.Config:
    return _stages(kimi_k3_debugmodel_c4(), _SIXTEEN_STAGES)


def kimi_k3_debugmodel_c4_16stages_naive() -> Trainer.Config:
    return _naive(_stages(kimi_k3_debugmodel_c4(), _SIXTEEN_STAGES))


def kimi_k3_debugmodel_c4_seed() -> Trainer.Config:
    from torchtitan.components.checkpointer import CheckpointManager

    config = kimi_k3_debugmodel_c4()
    config.checkpointer = CheckpointManager.Config()
    config.create_seed_checkpoint = True
    return config
