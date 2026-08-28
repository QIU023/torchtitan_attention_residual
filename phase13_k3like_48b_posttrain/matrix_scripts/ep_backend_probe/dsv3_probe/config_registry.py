# deepseek_v3 debugmodel with MinimalAsyncEP but the PLAIN GroupedExperts (no fused_swiglu override).
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.trainer import Trainer


def dsv3_std() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.activation_checkpoint = FullAC.Config()
    config.training.disable_cuda_graphs = True
    return config


def dsv3_maep_plain() -> Trainer.Config:
    config = dsv3_std()
    config.model_spec = model_registry("debugmodel", moe_comm_backend="minimal_async_ep")
    return config
