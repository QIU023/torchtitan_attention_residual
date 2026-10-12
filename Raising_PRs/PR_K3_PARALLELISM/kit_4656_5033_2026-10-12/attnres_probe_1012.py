"""Local probe recipes for PR 4656 rebased onto #5033 (list carrier, compiled aggregation); logbook kit only.

ac_cell: the Kimi K3 debug model on one GPU. PROBE_AC none / selective / full / region, PROBE_SEQ the sequence
length (default the recipe's), PROBE_STEPS the step count (default 20), PROBE_EAGER_AR=1 drops "attention_residual"
from the local compile regions (the aggregation runs eager, as before #5033). Deterministic, seed 42.
pp_cell: the B200 suite's 8-GPU cell (FSDP 2, TP 2, EP 2, PP 2, VPP 4), deterministic, seed 42, PROBE_STEPS steps.
"""

import os

from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def _common(config):
    config.training.steps = int(os.environ.get("PROBE_STEPS", "20"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    if os.environ.get("PROBE_EAGER_AR") == "1":
        config.model.local_compile_regions = [
            region for region in config.model.local_compile_regions if region != "attention_residual"
        ]
    return config


def ac_cell():
    seq = os.environ.get("PROBE_SEQ")
    config = kimi_k3_debugmodel(seq_len=int(seq)) if seq else kimi_k3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 1
    config.activation_checkpoint = {
        "none": None,
        "selective": SelectiveAC.Config(),
        "full": FullAC.Config(),
        "region": RegionAC.Config(save_regions=[]),
    }[os.environ.get("PROBE_AC", "none")]
    return _common(config)


def pp_cell():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4

    return _common(kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4())
