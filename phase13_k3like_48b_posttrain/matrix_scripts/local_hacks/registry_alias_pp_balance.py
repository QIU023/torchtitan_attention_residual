def kimi_k3_debugmodel_pp_balance() -> Trainer.Config:  # PROBE ONLY (not committed)
    import functools

    from torchtitan.models.kimi_k3.parallelize import pipeline_kimi_k3
    from torchtitan.models.kimi_k3.pp_balance import PPBalanceKnobs

    config = kimi_k3_debugmodel()
    assert config.model_spec is not None
    config.model_spec.pipelining_fn = functools.partial(
        pipeline_kimi_k3,
        pp_balance=PPBalanceKnobs(pp_balance_source_ranks=(0,), pp_balance_dest_rank=1),
    )
    return config
