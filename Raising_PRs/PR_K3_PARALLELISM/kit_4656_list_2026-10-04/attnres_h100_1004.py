"""H100 cells for PR 4656 (attention-residual blocks as a list) against main; logbook kit only, never committed.

cell(): Kimi K3 debug model with AdamW(lr=8e-4) (DistMuon rejects TP layouts), seed 42, deterministic. Env:
AR_STEPS (10), AR_TOKENS_STEP (256), AR_TOKENS_MB (256), AR_DP (1), AR_TP (1), AR_EP (1), AR_SP (1 = sequence
parallel on), AR_AC (none / selective / full / region; default selective, the flavor's).
mm_cell(): the B200 suite's kimi_k3_debugmodel_mm with type checking off, AR_STEPS steps.
"""

import os

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC

from torchtitan_recipes.tests import _set_spmd_typechecking


def _probe(config):
    config.training.steps = int(os.environ.get("AR_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def cell():
    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    e = os.environ
    config = kimi_k3_debugmodel()
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    config.training.num_tokens_per_train_step = int(e.get("AR_TOKENS_STEP", "256"))
    config.training.num_tokens_per_microbatch_per_dp_rank = int(e.get("AR_TOKENS_MB", "256"))
    config.parallelism.data_parallel_shard_degree = int(e.get("AR_DP", "1"))
    config.parallelism.tensor_parallel_degree = int(e.get("AR_TP", "1"))
    config.parallelism.expert_parallel_degree = int(e.get("AR_EP", "1"))
    config.parallelism.enable_sequence_parallel = e.get("AR_SP", "1") == "1"
    config.activation_checkpoint = {
        "none": None,
        "selective": SelectiveAC.Config(),
        "full": FullAC.Config(),
        "region": RegionAC.Config(save_regions=[]),
    }[e.get("AR_AC", "selective")]
    return _probe(config)


def mm_cell():
    from torchtitan_recipes.tests.suites.b200 import kimi_k3_debugmodel_mm

    config = kimi_k3_debugmodel_mm()
    _set_spmd_typechecking(config, typechecking=False)
    return _probe(config)
