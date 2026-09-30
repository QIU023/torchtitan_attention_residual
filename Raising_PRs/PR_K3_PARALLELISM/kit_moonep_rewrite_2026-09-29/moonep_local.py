"""Local recipes for tonight's smoke (never committed): the h100 MoonEP cell, and the
same cell on the standard and DeepEP backends. MOONEP_AC=none or full (09-30) replaces
the recipe's SelectiveAC, which fails on real MoonEP in the recompute."""

import os

from torchtitan.trainer import Trainer


def _ac(config: Trainer.Config) -> Trainer.Config:
    mode = os.environ.get("MOONEP_AC")
    if mode == "none":
        config.activation_checkpoint = None
    elif mode == "full":
        from torchtitan.distributed.activation_checkpoint import FullAC

        config.activation_checkpoint = FullAC.Config()
    return config


def moonep_cell() -> Trainer.Config:
    from torchtitan_recipes.tests.h100 import kimi_k3_moonep_fsdp4_ep4

    return _ac(kimi_k3_moonep_fsdp4_ep4())


def _with_backend(backend: str) -> Trainer.Config:
    from torchtitan.models.kimi_k3 import model_registry

    config = moonep_cell()
    config.model = model_registry(
        "debugmodel", enable_sp=True, seq_len=512, moe_comm_backend=backend
    )
    return config


def standard_cell() -> Trainer.Config:
    return _with_backend("standard")


def deepep_cell() -> Trainer.Config:
    return _with_backend("deepep")


def standard_ep2_cell() -> Trainer.Config:
    """The same data under another reduction order: EP 2 instead of 4, FSDP 4."""
    config = standard_cell()
    config.parallelism.expert_parallel_degree = 2
    return config
