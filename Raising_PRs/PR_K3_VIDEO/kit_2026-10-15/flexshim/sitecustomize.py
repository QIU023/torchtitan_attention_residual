# Local shim, never committed: this box has no FA4, so the K3 parity script builds TorchTitan with flex attention.
import os

if os.environ.get("K3_FLEX") == "1":
    import functools

    import torchtitan.models.kimi_k3 as _k3

    _orig = _k3.build_model_config

    @functools.wraps(_orig)
    def _build(*args, **kwargs):
        kwargs.setdefault("attn_backend", "flex")
        return _orig(*args, **kwargs)

    _k3.build_model_config = _build
