"""Measure gradients inside a REAL trainer run, varying only TP.

The hand-built probe could not reproduce the 3x grad_norm gap because it removed
every candidate cause: it dropped KDA (to run fp32), had no FSDP, and never went
through the trainer's parallel_dims. So this one changes nothing about the run --
it monkeypatches clip_grad_norm_ to report the MATERIALIZED global norm next to
the number the trainer prints, then delegates to torchtitan.train untouched.

Materialized means every gradient's full_tensor(), which reduces Partial and
gathers Shard placements. If the two numbers agree, the reported norm is right
and the TP/non-TP gap is a property of the gradients themselves. If they diverge
under TP but not without it, the reporting is wrong on the TP axis.

Usage: python -m ... tp_trainer_grad_probe.py <all the usual train args>
"""

from __future__ import annotations

import runpy
import sys

import torch


_DUMPED = []


def _param_names(params):
    """Recover parameter names by identity from the registered model parts."""
    mapping = {}
    models = _MODEL_PARTS[0] if _MODEL_PARTS else []
    by_id = {}
    for m in models:
        for n, p in m.named_parameters():
            by_id[id(p)] = n
    for i, p in enumerate(params):
        if id(p) in by_id:
            mapping[i] = by_id[id(p)]
    return mapping


_MODEL_PARTS = []


def _dump(per_param: dict) -> None:
    import json
    import os

    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if _DUMPED:
        return
    _DUMPED.append(1)
    path = os.environ.get("GRADCHK_DUMP")
    if path:
        with open(path, "w") as f:
            json.dump(per_param, f)
        print(f"[GRADCHK] per-parameter norms -> {path}", flush=True)


def _install() -> None:
    import torchtitan.distributed.utils as du

    original = du.clip_grad_norm_

    def patched(parameters, *args, **kwargs):
        params = list(parameters)
        # BEFORE the original call: clip_grad_norm_ rescales in place, so
        # measuring afterwards just reads back max_norm (1.0) and says nothing.
        total = 0.0
        n_dtensor = 0
        per_param = {}
        names = _param_names(params)
        with torch.no_grad():
            for i, p in enumerate(params):
                g = p.grad
                if g is None:
                    continue
                if hasattr(g, "full_tensor"):
                    n_dtensor += 1
                    g = g.full_tensor()
                sq = g.float().pow(2).sum().item()
                total += sq
                per_param[names.get(i, f"param{i}")] = sq**0.5
        materialized = total**0.5
        _dump(per_param)
        reported = original(params, *args, **kwargs)

        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            r = float(reported)
            print(
                f"[GRADCHK] reported={r:.6f}  materialized={materialized:.6f}  "
                f"ratio={materialized / max(r, 1e-12):.6f}  "
                f"dtensor_grads={n_dtensor}/{len(params)}",
                flush=True,
            )
        return reported

    du.clip_grad_norm_ = patched

    # Grab the model parts once the Trainer exists, so gradients can be labelled.
    import torchtitan.trainer as tt

    _orig_init = tt.Trainer.__init__

    def _init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        _MODEL_PARTS.clear()
        _MODEL_PARTS.append(self.model_parts)

    tt.Trainer.__init__ = _init
    # the trainer imported it as `dist_utils`, so patching the module attribute
    # is enough only if it resolves at call time -- it does (dist_utils.clip_...)


if __name__ == "__main__":
    _install()
    sys.argv[0] = "torchtitan.train"
    runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)
