"""Local recipe (never committed): the B200 mm cell with SPMD type checking off, so its TP and SP
boundary redistribution runs without the type assertions (main fails one on the EP axis here)."""

from torchtitan.trainer import Trainer


def mm_notc() -> Trainer.Config:
    from torchtitan_recipes.tests import _set_spmd_typechecking
    from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_mm

    config = kimi_k3_debugmodel_mm()
    _set_spmd_typechecking(config, typechecking=False)
    return config
