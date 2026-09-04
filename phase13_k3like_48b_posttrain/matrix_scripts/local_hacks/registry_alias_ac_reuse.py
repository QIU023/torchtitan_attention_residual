def kimi_k3_debugmodel_ac_reuse() -> Trainer.Config:  # PROBE ONLY (not committed)
    config = kimi_k3_debugmodel()
    assert config.model_spec is not None
    config.model_spec.model.ac_reuse_attention = True
    return config
