"""The B200 suite's DEP cell (FSDP 2, TP 2, EP 2, PP 2, VPP 4, vision_dep on), deterministic, seed 42, PROBE_STEPS steps."""

import os


def dep_cell():
    from torchtitan_recipes.tests.suites.b200 import kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4

    config = kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4()
    config.training.steps = int(os.environ.get("PROBE_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
