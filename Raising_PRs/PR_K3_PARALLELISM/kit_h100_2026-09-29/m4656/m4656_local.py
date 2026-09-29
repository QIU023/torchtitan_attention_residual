"""Local recipes for the #4656 TP/SP matrix (never committed). All cells use AdamW(lr=8e-4), as main's
TP recipes do (DistMuon rejects TP-produced _StridedShard storage) and as #4499's matrix ran."""

from torchtitan.components.optimizer import default_adamw
from torchtitan.trainer import Trainer


def k3_adamw() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    config.optimizer = default_adamw(lr=8e-4)
    return config


def k3_adamw_noac() -> Trainer.Config:
    config = k3_adamw()
    config.activation_checkpoint = None
    return config


def k3_adamw_typecheck() -> Trainer.Config:
    from torchtitan_recipes.tests import _set_spmd_typechecking

    config = k3_adamw()
    _set_spmd_typechecking(config, typechecking=True)
    return config


def k3_mm() -> Trainer.Config:
    from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_mm

    return kimi_k3_debugmodel_mm()


def k3_mm_notc() -> Trainer.Config:
    from torchtitan_recipes.tests import _set_spmd_typechecking

    config = k3_mm()
    _set_spmd_typechecking(config, typechecking=False)
    return config
