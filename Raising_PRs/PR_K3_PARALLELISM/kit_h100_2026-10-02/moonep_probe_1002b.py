"""Local probe recipes for the MoonEP refactor check (logbook kit only, never committed to a branch).

The cell is the one the removed h100 recipe kimi_k3_moonep_fsdp4_ep4 built (Kimi K3 debug model, seq 512, FSDP 4 x EP 4,
the recipe's selective activation checkpointing), built here so that it runs on both trees: MoonEPRoutedExperts lives
in models/common/moe.py before the refactor and in distributed/moonep/experts.py after it. PROBE_STEPS sets the step
count (default 100); the runs are deterministic with seed 42.
"""

import os
from dataclasses import replace

from torchtitan.config.transform import apply_transforms, TokenDispatcherTransform
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

try:
    from torchtitan.distributed.moonep.experts import MoonEPRoutedExperts
except ImportError:
    from torchtitan.models.common.moe import MoonEPRoutedExperts


def moonep_cell():
    config = kimi_k3_debugmodel(seq_len=512)
    config = replace(
        config,
        parallelism=replace(config.parallelism, expert_parallel_degree=4),
    )
    config.parallelism.data_parallel_shard_degree = 4
    config = apply_transforms(
        config,
        [TokenDispatcherTransform(dispatcher=MoonEPTokenDispatcher, routed_experts=MoonEPRoutedExperts)],
    )
    config.training.steps = int(os.environ.get("PROBE_STEPS", "100"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
