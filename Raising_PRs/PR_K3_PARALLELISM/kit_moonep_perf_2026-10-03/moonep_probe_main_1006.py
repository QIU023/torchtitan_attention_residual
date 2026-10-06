"""The standard-EP cell of moonep_probe_1003.py for a tree without MoonEP (plain main); logbook kit only."""

import os
from dataclasses import replace

from torchtitan.config.transform import apply_transforms

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def standard_cell():
    config = kimi_k3_debugmodel(seq_len=512)
    config = replace(config, parallelism=replace(config.parallelism, expert_parallel_degree=4))
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.data_parallel_replicate_degree = 1
    config = apply_transforms(config, [])
    config.training.steps = int(os.environ.get("PROBE_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = os.environ.get("PROBE_DET", "1") == "1"
    return config
