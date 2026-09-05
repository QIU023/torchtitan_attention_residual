def kimi_k3_debugmodel_tc() -> Trainer.Config:  # PROBE ONLY (not committed)
    """The debug flavor the way 4446's B200 cell runs it: spmd_types with typechecking, AC off."""
    from torchtitan_recipes.tests import _use_spmd_types

    config = kimi_k3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    return config
