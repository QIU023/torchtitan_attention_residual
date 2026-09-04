def kimi_k3_debugmodel_cc12m() -> Trainer.Config:  # PROBE ONLY (not committed)
    """The debug model on the streamed cc12m (no sample repeats in 100 steps; the 32-sample
    test set is memorized by step 90)."""
    config = kimi_k3_debugmodel()
    config.dataloader = _kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m"])
    return config


def kimi_k3_debugmodel_cc12m_pp_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    import functools

    from torchtitan.models.kimi_k3.parallelize import pipeline_kimi_k3

    config = kimi_k3_debugmodel_cc12m()
    assert config.model_spec is not None
    config.model_spec.pipelining_fn = functools.partial(
        pipeline_kimi_k3, attn_res_cache=False
    )
    return config
