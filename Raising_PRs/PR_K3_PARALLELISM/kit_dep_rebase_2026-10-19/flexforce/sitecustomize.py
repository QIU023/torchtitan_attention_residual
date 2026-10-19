"""Local shim, never committed: this box has no flash_attn, so K3 recipes build with flex attention when K3_FORCE_FLEX=1."""
import os
import sys

if os.environ.get("K3_FORCE_FLEX") == "1":
    import functools

    import torchtitan.models.kimi_k3 as _k3

    _orig = _k3.build_model_config

    @functools.wraps(_orig)
    def _build(*args, **kwargs):
        kwargs["attn_backend"] = "flex"
        return _orig(*args, **kwargs)

    _k3.build_model_config = _build
    sys.stderr.write(f"K3_FORCE_FLEX active in pid {os.getpid()}\n")
