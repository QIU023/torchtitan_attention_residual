"""Local probe recipes for the MoonEP recheck on the rebased tree (logbook kit only, never committed to a branch).

The cells mirror the h100 suite's kimi_k3_moonep_fsdp4_ep4 (Kimi K3 debug model, seq 512, FSDP 4 x EP 4); the
standard cells build the same config without the MoonEP transform. PROBE_STEPS sets the step count (default 20),
PROBE_DET=0 turns deterministic mode off (timing cells).
"""

import os
from dataclasses import replace

from torchtitan.config.transform import apply_transforms
from torchtitan.distributed.activation_checkpoint import FullAC

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel
from torchtitan_recipes.tests.suites.h100 import kimi_k3_moonep_fsdp4_ep4


def _probe(config):
    config.training.steps = int(os.environ.get("PROBE_STEPS", "20"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = os.environ.get("PROBE_DET", "1") == "1"
    return config


def _standard(ep: int):
    config = kimi_k3_debugmodel(seq_len=512)
    config = replace(
        config,
        parallelism=replace(config.parallelism, expert_parallel_degree=ep),
    )
    config.parallelism.data_parallel_shard_degree = 4
    return apply_transforms(config, [])


def standard_cell():
    return _probe(_standard(4))


def standard_ep2_cell():
    return _probe(_standard(2))


def moonep_cell():
    return _probe(kimi_k3_moonep_fsdp4_ep4())


def moonep_full_cell():
    config = kimi_k3_moonep_fsdp4_ep4()
    config.activation_checkpoint = FullAC.Config()
    return _probe(config)
