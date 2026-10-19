"""Local kit, never committed: the B200 fsdp2 x tp2 x ep2 x pp2 x vpp4 cell, seeded and deterministic from one seed
checkpoint (DEP_SEED), with DEP_OFF=1 turning vision_dep off where the field exists; seed() writes the checkpoint."""
import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import ParallelismConfig


def cell():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4

    config = kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4()
    config.debug.seed = 42
    config.debug.deterministic = True
    config.training.steps = int(os.environ.get("DEP_STEPS", "10"))
    if os.environ.get("DEP_OFF") == "1" and hasattr(config.model, "vision_dep"):
        config.model.vision_dep.enabled = False
    config.checkpointer = CheckpointManager.Config(
        interval=100000, initial_load_path=os.environ["DEP_SEED"], initial_load_model_only=True
    )
    return config


def seed():
    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    from torchtitan.components.optim import AdamW, OptimizersContainer

    config = kimi_k3_debugmodel(parallelism=ParallelismConfig())
    config.debug.seed = 42
    config.optim.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    config.checkpointer = CheckpointManager.Config(folder=os.environ["DEP_SEED_OUT"])
    config.create_seed_checkpoint = True
    config.training.steps = 1
    return config
