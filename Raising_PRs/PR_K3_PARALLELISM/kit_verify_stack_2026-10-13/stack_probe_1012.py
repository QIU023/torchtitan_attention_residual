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


def _mxqat(config):
    from typing import cast

    from torchtitan.config.transform import MXQATTransform
    from torchtitan.models.kimi_k3.model import KimiK3Model
    from torchtitan.models.kimi_k3.quantization import MXFP4_QUANTIZATION_CONFIG
    from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
    from torchtitan.quantization.mx_qat.checkpoint import MXFP4CheckpointPolicy

    adapter = KimiK3StateDictAdapter(cast(KimiK3Model.Config, config.model), hf_assets_path=None)
    policy = MXFP4CheckpointPolicy.from_config(MXFP4_QUANTIZATION_CONFIG, adapter.hf_linear_weight_mapping())
    return MXQATTransform.from_weight_fqns(config.model, adapter.qat_weight_fqns(policy))


def _with(config, *names):
    from torchtitan.config.transform import apply_transforms, LoRATransform

    transforms = []
    if "mxqat" in names:
        transforms.append(_mxqat(config))
    if "lora" in names:
        transforms.append(LoRATransform(rank=8, alpha=16.0))
    return _common(apply_transforms(config, transforms))


def mm_mxqat():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_mm

    return _with(kimi_k3_debugmodel_mm(), "mxqat")


def mm_lora():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_mm

    return _with(kimi_k3_debugmodel_mm(), "lora")


def mm_qlora():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_mm

    return _with(kimi_k3_debugmodel_mm(), "mxqat", "lora")


def mm_muon_mxqat():
    from tests.integration_tests.b200 import kimi_k3_debugmodel_mm_muon

    return _with(kimi_k3_debugmodel_mm_muon(), "mxqat")
