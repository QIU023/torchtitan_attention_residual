"""Local recipes (never committed): the MoonEP h100 cell's layout on the standard dispatcher, and both at a larger per-rank token count."""

import os

from torchtitan.trainer import Trainer


def std_fsdp4_ep4() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(seq_len=int(os.environ.get("MEP_SEQ", "512")))
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 4
    return config


def moonep_fsdp4_ep4() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_moonep

    config = kimi_k3_debugmodel_moonep(seq_len=int(os.environ.get("MEP_SEQ", "512")))
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 4
    return config


def deepep_fsdp4_ep4() -> Trainer.Config:
    from torchtitan.models.kimi_k3 import model_registry
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    seq = int(os.environ.get("MEP_SEQ", "512"))
    config = kimi_k3_debugmodel(seq_len=seq)
    config.model_spec = model_registry("debugmodel", seq_len=seq, moe_comm_backend="deepep")
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 4
    return config
