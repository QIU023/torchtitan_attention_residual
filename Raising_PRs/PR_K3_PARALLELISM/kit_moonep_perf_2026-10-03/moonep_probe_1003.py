"""Local probe recipes for the MoonEP performance round (logbook kit only, never committed to a branch).

Kimi K3 debug model, seq 512, FSDP 4 x EP 4 (the removed h100 cell's shape); the *_ep2 and *_hsdp cells change the layout. PROBE_STEPS sets the step count
(default 10), PROBE_DET=0 turns deterministic mode off (timing cells), PROBE_PROFILE=1 writes a profiler trace
of steps PROBE_PROFILE_STEPS (default 6,7). The shared-stream cell needs the review branch after 09cf5783c.
"""

import os
from dataclasses import replace

from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.token_dispatcher import TokenDispatcherTransform
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def _probe(config):
    config.training.steps = int(os.environ.get("PROBE_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = os.environ.get("PROBE_DET", "1") == "1"
    if os.environ.get("PROBE_PROFILE", "0") == "1":
        config.profiler.enable_profiling = True
        first, last = (int(s) for s in os.environ.get("PROBE_PROFILE_STEPS", "6,7").split(","))
        config.profiler.profile_freq = last + 1
        config.profiler.profiler_warmup = 0
        config.profiler.profiler_active = last - first + 1
    return config


def _base(ep: int = 4, dp_shard: int = 4, dp_replicate: int = 1):
    config = kimi_k3_debugmodel(seq_len=512)
    config = replace(config, parallelism=replace(config.parallelism, expert_parallel_degree=ep))
    config.parallelism.data_parallel_shard_degree = dp_shard
    config.parallelism.data_parallel_replicate_degree = dp_replicate
    return config


def _moonep(config=None):
    from torchtitan.distributed.moonep.experts import MoonEPRoutedExperts

    return apply_transforms(
        _base() if config is None else config,
        [TokenDispatcherTransform(dispatcher=MoonEPTokenDispatcher, routed_experts=MoonEPRoutedExperts)],
    )


def _shared_stream(config):
    for _, moe, _, _ in config.model.traverse(MoE.Config):
        moe.shared_experts_stream = True
    return config


def standard_cell():
    return _probe(apply_transforms(_base(), []))


def standard_shared_stream_cell():
    return _probe(_shared_stream(apply_transforms(_base(), [])))


def moonep_cell():
    return _probe(_moonep())


def moonep_shared_stream_cell():
    return _probe(_shared_stream(_moonep()))


def moonep_full_cell():
    config = _moonep()
    config.activation_checkpoint = FullAC.Config()
    return _probe(config)


# Expert FSDP over two ranks (dp_shard 4 x EP 2) and HSDP (dp_replicate 2 x dp_shard 2 x EP 2), item 12 of 10-04.
def standard_ep2_cell():
    return _probe(apply_transforms(_base(ep=2), []))


def moonep_ep2_cell():
    return _probe(_moonep(_base(ep=2)))


def standard_hsdp_cell():
    return _probe(apply_transforms(_base(ep=2, dp_shard=2, dp_replicate=2), []))


def moonep_hsdp_cell():
    return _probe(_moonep(_base(ep=2, dp_shard=2, dp_replicate=2)))
