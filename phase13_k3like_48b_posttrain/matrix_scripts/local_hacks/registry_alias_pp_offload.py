def kimi_k3_debugmodel_pp_offload() -> Trainer.Config:  # PROBE ONLY (not committed)
    import functools

    from torchtitan.models.kimi_k3.parallelize import pipeline_kimi_k3

    config = kimi_k3_debugmodel()
    assert config.model_spec is not None
    config.model_spec.pipelining_fn = functools.partial(
        pipeline_kimi_k3, attn_res_cache_offload=True
    )
    return config
