# Added to torchtitan/models/kimi_k3/config_registry.py on the H100 box AFTER
# onbox_local_flavors_and_guard_lift.patch was taken, so they are kept here
# rather than in that patch. They need _strip_quantile_balancing, which the
# patch does carry.


def kimi_k3_debugmodel_noqb_c4(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    """C4 with quantile balancing off (local flavor, not for upstream)."""
    return _strip_quantile_balancing(kimi_k3_debugmodel_c4(seq_len=seq_len))


def kimi_k3_debugmodel_noqb_moonep_c4(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    """C4, MoonEP, quantile balancing off (local flavor, not for upstream)."""
    return _strip_quantile_balancing(kimi_k3_debugmodel_moonep_c4(seq_len=seq_len))
