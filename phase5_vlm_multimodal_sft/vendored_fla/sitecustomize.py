"""Shadow buggy fla modules with our PR13-series patched copies.

Python auto-imports `sitecustomize` at interpreter startup if it is importable
on sys.path. This dir is placed on PYTHONPATH by run_seqkd_sft_autoresume.sh,
so every torchrun worker runs this and picks up the patches — WITHOUT touching
the installed fla package (site-packages stays pristine).

Each entry maps a fully-qualified fla module name to our patched copy. A single
MetaPathFinder at sys.meta_path[0] intercepts the import of those exact module
names and loads our patched file under the same name (so its relative/absolute
fla imports resolve normally).

Fixes (all the same Blackwell sm_120 + Triton 3.6.0 bug class — phantom autotuner
keys forcing re-autotune that crashes, and/or BS<BT make_block_ptr overlap;
mirrors upstream fla #796):
  - fla.modules.fused_norm_gate         (PR13  — MLA o_norm gated RMSNorm)
  - fla.modules.conv.triton.kernels     (PR13b — KDA short-conv causal_conv1d)
See Raising_PRs/PR13*/ for per-fix diagnosis + git-apply-able patches.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))

# fully-qualified module name -> patched file in this dir
# NOTE: causal_conv1d (conv.triton.kernels) and l2norm "drop phantom NB key"
# patches were REVERTED 2026-05-30 — live A/B showed they REGRESSED the conv
# crash (old kernel ~339 steps/crash → patched ~30-99 steps/crash). The NB key,
# though unused in the body, partitions the autotune cache by T-bucket; dropping
# it reuses one (unsafe-for-some-T) config across all T. The device-side assert
# is a genuine data-dependent in-kernel OOB, NOT the autotuner — needs a GPU
# repro to root-cause (deferred to post-seq-KD when GPUs are free).
# fused_norm_gate (PR13) kept: original, addresses the rarer ~2500-step crash,
# ran ~500 steps without an observed regression.
_SHADOWS = {
    "fla.modules.fused_norm_gate": "fused_norm_gate_patched.py",
}


class _FlaShim(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        rel = _SHADOWS.get(fullname)
        if rel is None:
            return None
        patched = os.path.join(_DIR, rel)
        if not os.path.isfile(patched):
            return None
        return importlib.util.spec_from_file_location(fullname, patched)


def _install():
    # Don't shadow a module that's already imported (we must win before first
    # import); install the finder only if not already present.
    if any(isinstance(f, _FlaShim) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _FlaShim())
    if os.environ.get("FLA_SHIM_VERBOSE"):
        already = [m for m in _SHADOWS if m in sys.modules]
        for m, rel in _SHADOWS.items():
            tag = " (ALREADY IMPORTED — shadow inactive)" if m in sys.modules else ""
            print(f"[fla-shim] will redirect {m} -> {rel}{tag}", flush=True)
        if already:
            print(f"[fla-shim] WARNING already-imported (shadow missed): {already}", flush=True)


try:
    _install()
except Exception as e:  # never break interpreter startup
    if os.environ.get("FLA_SHIM_VERBOSE"):
        print(f"[fla-shim] install failed: {e}", flush=True)
