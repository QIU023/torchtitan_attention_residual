"""Probe recipes for the k3_mxqat_stack_20261012 full run, logbook kit only: the Kimi K3 CI cells and the MX QAT
recipe with seed 42, deterministic, log every step, PROBE_STEPS steps (default 20)."""

import os


def _common(config):
    config.training.steps = int(os.environ.get("PROBE_STEPS", "20"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def mm():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_mm

    return _common(kimi_k3_debugmodel_mm())


def mm_muon():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_mm_muon

    return _common(kimi_k3_debugmodel_mm_muon())


def pp_vpp4():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4

    return _common(kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4())


def cp_allgather():
    from tests.integration_tests.h100 import kimi_k3_debugmodel_mm_allgather_kv_cp2

    return _common(kimi_k3_debugmodel_mm_allgather_kv_cp2())


def cp_ulysses():
    from tests.integration_tests.h100 import kimi_k3_debugmodel_mm_ulysses_cp2

    return _common(kimi_k3_debugmodel_mm_ulysses_cp2())


def mx_qat():
    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel_mx_qat

    return _common(kimi_k3_debugmodel_mx_qat())
