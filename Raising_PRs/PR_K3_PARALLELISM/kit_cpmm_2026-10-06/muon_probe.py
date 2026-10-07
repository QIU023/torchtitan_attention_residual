"""Kimi K3 recipes with DistMuon kept, seeded and deterministic, for main-vs-fix equality of the CP Muon layout fix;
logbook kit only. Env: MP_RECIPE debugmodel | agcp | ulcp (the h100 CP cells) [debugmodel]; MP_STEPS [10]."""

import os


def cell():
    recipe = os.environ.get("MP_RECIPE", "debugmodel")
    if recipe == "debugmodel":
        from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

        config = kimi_k3_debugmodel()
    else:
        from torchtitan_recipes.tests.suites import h100

        config = {
            "agcp": h100.kimi_k3_debugmodel_mm_allgather_kv_cp2,
            "ulcp": h100.kimi_k3_debugmodel_mm_ulysses_cp2,
        }[recipe]()
    config.training.steps = int(os.environ.get("MP_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
