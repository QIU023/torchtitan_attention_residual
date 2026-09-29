"""Local recipes (never committed): the MoonEP h100 cell and the same cell on the standard backend."""

from torchtitan.trainer import Trainer


def moonep_cell() -> Trainer.Config:
    from torchtitan_recipes.tests.h100 import kimi_k3_moonep_fsdp4_ep4

    return kimi_k3_moonep_fsdp4_ep4()


def standard_cell() -> Trainer.Config:
    from torchtitan.models.kimi_k3 import model_registry

    config = moonep_cell()
    config.model = model_registry(
        "debugmodel", enable_sp=True, seq_len=512, moe_comm_backend="standard"
    )
    return config
