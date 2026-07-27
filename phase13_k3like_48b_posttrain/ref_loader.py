"""Load Kimi K3's reference modeling code under our transformers version.

The reference targets a transformers release whose ``utils.generic`` still
exported OutputRecorder / check_model_inputs; ours (5.14.1) does not. Only the
PreTrainedModel plumbing needs them, and we want the plain nn.Module classes, so
stub whatever is missing rather than pinning a second transformers.
"""

import importlib.util
import pathlib
import sys

REF_DIR = pathlib.Path(
    "/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain"
    "/official_k3/reference"
)


def _shim_transformers() -> list[str]:
    import transformers.utils.generic as g

    added = []

    class OutputRecorder:  # noqa: D401 - stub
        def __init__(self, *a, **kw):
            pass

    def check_model_inputs(*a, **kw):
        def deco(fn):
            return fn

        return deco if not (a and callable(a[0])) else a[0]

    for name, obj in (
        ("OutputRecorder", OutputRecorder),
        ("check_model_inputs", check_model_inputs),
    ):
        if not hasattr(g, name):
            setattr(g, name, obj)
            added.append(name)
    return added


_PKG = "k3ref"


def _ensure_package() -> None:
    """Register REF_DIR as a package so the files' relative imports resolve."""
    if _PKG in sys.modules:
        return
    import types

    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(REF_DIR)]
    sys.modules[_PKG] = pkg


def load(name: str):
    """Import a reference file by stem, e.g. load('modeling_kimi_linear')."""
    _shim_transformers()
    _ensure_package()
    full = f"{_PKG}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, REF_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    for f in ("configuration_kimi_k3", "modeling_kimi_linear", "modeling_kimi_k3"):
        try:
            m = load(f)
            names = [n for n in dir(m) if n[0].isupper()][:8]
            print(f"OK   {f}: {names}")
        except Exception as e:
            print(f"FAIL {f}: {type(e).__name__}: {e}")
