"""Shadow fla's buggy `fla.modules.fused_norm_gate` with our PR13-patched copy.

Python auto-imports `sitecustomize` at interpreter startup if it is importable
on sys.path. This dir is placed on PYTHONPATH by run_seqkd_sft_autoresume.sh,
so every torchrun worker runs this and picks up the patch — WITHOUT touching
the installed fla package (site-packages stays pristine).

The patch fixes the fla `fused_norm_gate.py` Blackwell sm_120 KDA crash:
phantom `NB` autotuner key (forces re-autotuning → autotuner crash) + `BS < BT`
overlapping writes on `dx`. See Raising_PRs/PR13_fla_fused_norm_gate_sm120_kda_crash/.

Mechanism: a MetaPathFinder inserted at sys.meta_path[0] intercepts the import
of exactly `fla.modules.fused_norm_gate` and loads our patched file under that
fully-qualified name (so its relative/absolute fla imports resolve normally).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_TARGET = "fla.modules.fused_norm_gate"
_PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fused_norm_gate_patched.py")


class _FusedNormGateShim(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        if not os.path.isfile(_PATCHED):
            return None
        return importlib.util.spec_from_file_location(fullname, _PATCHED)


# Only install if fla is the version this patch was written against, and the
# target isn't already imported (we must win before first import).
def _install():
    if _TARGET in sys.modules:
        return  # too late; leave the installed module in place
    if any(isinstance(f, _FusedNormGateShim) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _FusedNormGateShim())
    if os.environ.get("FLA_SHIM_VERBOSE"):
        print(f"[fla-shim] redirecting {_TARGET} -> {_PATCHED}", flush=True)


try:
    _install()
except Exception as e:  # never break interpreter startup
    if os.environ.get("FLA_SHIM_VERBOSE"):
        print(f"[fla-shim] install failed: {e}", flush=True)
