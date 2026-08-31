# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EP comm-backend cells, as a recipe module rather than model flavors.

The backend used to be a flavor (kimi_k3_debugmodel_deepep and friends); those
were dropped because a flavor per backend is not how the registry expects to be
parameterized, so the cells select it through model_registry instead. Drop this
file into torchtitan_recipes/ on the branch under test and name it with
--module torchtitan_recipes.k3_ep_backends.

minimal_async_ep additionally requires full recompute: pass
``activation-checkpoint:full`` or it raises at startup.
"""

from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
from torchtitan.trainer import Trainer


def _backend(name: str) -> Trainer.Config:
    config = kimi_k3_debugmodel()
    config.model_spec = model_registry("debugmodel", moe_comm_backend=name)
    return config


def k3_standard() -> Trainer.Config:
    return _backend("standard")


def k3_deepep() -> Trainer.Config:
    return _backend("deepep")


def k3_minimal_async_ep() -> Trainer.Config:
    return _backend("minimal_async_ep")
