"""Scratch recipes for the dynamic CP rebase: the b200 mm CP cells with the tower forced replicated."""
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.b200 import (
    kimi_k3_debugmodel_mm_allgather_kv_cp2,
    kimi_k3_debugmodel_mm_ulysses_cp2,
)


def _replicated(config: Trainer.Config) -> Trainer.Config:
    assert config.model_spec is not None
    config.model_spec.model.dynamic_cp_min_patches = 10**9
    return config


def kimi_k3_mm_allgather_kv_cp2_replicated() -> Trainer.Config:
    return _replicated(kimi_k3_debugmodel_mm_allgather_kv_cp2())


def kimi_k3_mm_ulysses_cp2_replicated() -> Trainer.Config:
    return _replicated(kimi_k3_debugmodel_mm_ulysses_cp2())


def _min96(config: Trainer.Config) -> Trainer.Config:
    # The debug batch's image is 16 x 12 = 192 patches, under the default 256: lower the bar so it is cut.
    assert config.model_spec is not None
    config.model_spec.model.dynamic_cp_min_patches = 96
    return config


def kimi_k3_mm_allgather_kv_cp2_min96() -> Trainer.Config:
    return _min96(kimi_k3_debugmodel_mm_allgather_kv_cp2())


def kimi_k3_mm_ulysses_cp2_min96() -> Trainer.Config:
    return _min96(kimi_k3_debugmodel_mm_ulysses_cp2())


def _cp2_pp2(config: Trainer.Config) -> Trainer.Config:
    # dp1 x cp2 x pp2 on four GPUs; the K3 split places the tower with the embedding on stage 0.
    # SPMD type checking refuses a pipeline, so the b200 recipe's setting is undone.
    from torchtitan_recipes.tests.b200 import _set_spmd_typechecking

    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.num_pp_microbatches = 2
    return config


def _cp2_tp2(config: Trainer.Config) -> Trainer.Config:
    # dp1 x cp2 x tp2 on four GPUs; CP refuses sequence parallel, so it is off.
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = False
    return config


def kimi_k3_mm_allgather_kv_cp2_pp2_min96() -> Trainer.Config:
    return _cp2_pp2(_min96(kimi_k3_debugmodel_mm_allgather_kv_cp2()))


def kimi_k3_mm_allgather_kv_cp2_tp2_min96() -> Trainer.Config:
    return _cp2_tp2(_min96(kimi_k3_debugmodel_mm_allgather_kv_cp2()))


def kimi_k3_mm_ulysses_cp2_pp2_min96() -> Trainer.Config:
    return _cp2_pp2(_min96(kimi_k3_debugmodel_mm_ulysses_cp2()))


def kimi_k3_mm_ulysses_cp2_tp2_min96() -> Trainer.Config:
    return _cp2_tp2(_min96(kimi_k3_debugmodel_mm_ulysses_cp2()))
