"""Local cells for #4380 (dynamic CP of the Kimi K3 vision tower); logbook kit only, never committed.

cell(): main's kimi_k3_debugmodel_mm_allgather_kv_cp2 recipe. Env: CP_DEGREE (2), CP_DP (1, dp_shard),
CP_MINP (dynamic_cp_min_patches; 128 splits cc12m-test's 192-patch images, 100000 keeps them whole),
CP_TYPECHECK (0/1), CP_STEPS (10); seed 42, deterministic.
"""

import os

from torchtitan_recipes.tests import _set_spmd_typechecking


def cell():
    from torchtitan_recipes.tests.suites.h100 import kimi_k3_debugmodel_mm_allgather_kv_cp2

    e = os.environ
    config = kimi_k3_debugmodel_mm_allgather_kv_cp2()
    _set_spmd_typechecking(config, typechecking=e.get("CP_TYPECHECK", "0") == "1")
    config.parallelism.context_parallel_degree = int(e.get("CP_DEGREE", "2"))
    config.parallelism.data_parallel_shard_degree = int(e.get("CP_DP", "1"))
    assert config.model.vision_encoder is not None
    config.model.vision_encoder.dynamic_cp_min_patches = int(e.get("CP_MINP", "128"))
    config.training.steps = int(e.get("CP_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
