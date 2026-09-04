def kimi_k3_debugmodel_pp_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    import functools

    from torchtitan.models.kimi_k3.parallelize import pipeline_kimi_k3  # pp_review3 round 2: the entry lives in parallelize.py

    config = kimi_k3_debugmodel()
    assert config.model_spec is not None
    config.model_spec.pipelining_fn = functools.partial(
        pipeline_kimi_k3, attn_res_cache=False
    )
    return config
