"""Local probe recipes for PR 4656 (attention residual blocks as a list), logbook kit only.

Kimi K3 debug model on one GPU (dp 1). PROBE_AC selects none / selective / full / region activation
checkpointing, PROBE_SEQ the sequence length (default the recipe's), PROBE_STEPS the step count (default 10).
Deterministic, seed 42.
"""

import os

from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def ac_cell():
    seq = os.environ.get("PROBE_SEQ")
    config = kimi_k3_debugmodel(seq_len=int(seq)) if seq else kimi_k3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 1
    config.training.steps = int(os.environ.get("PROBE_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    config.activation_checkpoint = {
        "none": None,
        "selective": SelectiveAC.Config(),
        "full": FullAC.Config(),
        "region": RegionAC.Config(save_regions=[]),
    }[os.environ.get("PROBE_AC", "none")]
    return config
