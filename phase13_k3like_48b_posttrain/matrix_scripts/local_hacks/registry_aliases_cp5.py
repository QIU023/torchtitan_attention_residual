# PROBE ONLY (not committed): registry aliases for the recipe flavors and the generic kernels.
def kimi_k3_debugmodel_cp2() -> Trainer.Config:
    from torchtitan_recipes.tests.features import kimi_k3_debugmodel_cp2 as flavor

    return flavor()


def kimi_k3_debugmodel_cp2_allgather() -> Trainer.Config:
    from torchtitan_recipes.tests.features import (
        kimi_k3_debugmodel_cp2_allgather as flavor,
    )

    return flavor()


def kimi_k3_debugmodel_cp2_generic() -> Trainer.Config:
    from torchtitan.models.common.cp_attention import UlyssesCPFlexAttention
    from torchtitan_recipes.kimi_k3 import kimi_k3_context_parallel

    return kimi_k3_context_parallel(
        kimi_k3_debugmodel_mm(), cp_degree=2, mla_kernel=UlyssesCPFlexAttention
    )


def kimi_k3_debugmodel_cp2_allgather_generic() -> Trainer.Config:
    from torchtitan.models.common.cp_attention import AllGatherCPFlexAttention
    from torchtitan_recipes.kimi_k3 import kimi_k3_context_parallel

    return kimi_k3_context_parallel(
        kimi_k3_debugmodel_mm(), cp_degree=2, mla_kernel=AllGatherCPFlexAttention
    )
