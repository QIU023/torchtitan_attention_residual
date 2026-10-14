"""MoonEP and standard cells on the stack (logbook kit only): Kimi K3 debug model, seq 512, FSDP 4 x EP 4, PROBE_STEPS steps.
The model config is built with EP 4, as main requires since PR-4908."""

import os

from torchtitan.config import ParallelismConfig
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.token_dispatcher import TokenDispatcherTransform
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def _base():
    config = kimi_k3_debugmodel(seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=4))
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.data_parallel_replicate_degree = 1
    return config


def _probe(config):
    config.training.steps = int(os.environ.get("PROBE_STEPS", "2"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def moonep_cell():
    from torchtitan.distributed.moonep.experts import MoonEPRoutedExperts

    config = _base()
    tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    transform = TokenDispatcherTransform(
        dispatcher=MoonEPTokenDispatcher, routed_experts=MoonEPRoutedExperts, num_max_tokens_per_rank=tokens
    )
    return _probe(apply_transforms(config, [transform]))


def standard_cell():
    return _probe(apply_transforms(_base(), []))
