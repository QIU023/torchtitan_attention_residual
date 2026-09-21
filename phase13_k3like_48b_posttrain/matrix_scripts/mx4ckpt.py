"""Matrix wrapper flavors for main after the checkpoint component move (local only, never committed).

`--module mx4ckpt --config cell` builds `$MX4_BASE_MODULE` / `$MX4_BASE_CONFIG` with a load-only checkpointer;
`--config seed` builds the same flavor as a seed-checkpoint writer. The base module resolves as the CLI does.
"""

import importlib
import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.trainer import Trainer


def _registry():
    name = os.environ["MX4_BASE_MODULE"]
    for candidate in (
        f"torchtitan.models.{name}.config_registry",
        f"{name}.config_registry",
        name,
    ):
        try:
            return importlib.import_module(candidate)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(name)


def _base() -> Trainer.Config:
    return getattr(_registry(), os.environ["MX4_BASE_CONFIG"])()


def cell() -> Trainer.Config:
    config = _base()
    config.checkpointer = CheckpointManager.Config(interval=100000)
    return config


def seed() -> Trainer.Config:
    config = _base()
    config.checkpointer = CheckpointManager.Config()
    config.create_seed_checkpoint = True
    return config
