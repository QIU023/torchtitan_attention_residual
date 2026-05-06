"""DIAGNOSTIC: monkey-patch stage_backward to pass retain_graph=True.

Goal: tell apart two failure classes for the V>=2 + LBS>=3 + AttnRes
"Trying to backward through the graph a second time" crash:

  (A) Same grad_fn node IS reachable twice from this stage's backward
      roots within a SINGLE autograd.backward call -> retain_graph=True
      makes it pass.

  (B) Saved tensor's storage was overwritten or freed by another
      microbatch's prior actions (alias) -> retain_graph=True still
      fails (likely with "saved variable modified" or similar).

Activated only when env var ATTNRES_DIAG_RETAIN_GRAPH=1, otherwise no-op.
Idempotent.

Usage: imported by phase5/train_mm.py once the existing hotfix is in
place. The two patches are orthogonal.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.distributed.pipelining._backward as _bwd_mod

logger = logging.getLogger(__name__)

_PATCHED_FLAG = "_attnres_diag_retain_graph_applied"


def apply_retain_graph_diag() -> None:
    if os.environ.get("ATTNRES_DIAG_RETAIN_GRAPH") != "1":
        return
    if getattr(_bwd_mod, _PATCHED_FLAG, False):
        return

    original_stage_backward = _bwd_mod.stage_backward

    def patched_stage_backward(
        stage_output, output_grads, input_values, outputs_with_grads_idxs=None,
    ):
        # Re-implement the original body but pass retain_graph=True to
        # autograd.backward. Keeps the rest of the logic identical so any
        # bisect on the second run is meaningful.
        if outputs_with_grads_idxs is not None:
            stage_output = [stage_output[i] for i in outputs_with_grads_idxs]
            output_grads = [output_grads[i] for i in outputs_with_grads_idxs]

        try:
            stage_output_tensors = []
            output_grad_tensors = []

            def extract(ov, gv, fn):
                if isinstance(ov, torch.Tensor):
                    if not ov.requires_grad and ov.grad_fn is None:
                        return
                    stage_output_tensors.append(ov)
                    output_grad_tensors.append(gv)
                elif isinstance(ov, (tuple, list)):
                    if gv is None:
                        return
                    for o, g in zip(ov, gv):
                        fn(o, g, fn)

            extract(stage_output, output_grads, extract)

            torch.autograd.backward(
                stage_output_tensors,
                grad_tensors=output_grad_tensors,
                retain_graph=True,  # <<< THE DIAGNOSTIC FLAG
            )

            grad_inputs = []
            for val in input_values:
                if isinstance(val, torch.Tensor):
                    grad_inputs.append(val.grad)
                    val.grad = None
                else:
                    grad_inputs.append(None)
        except Exception as e:
            from torch.distributed.pipelining._backward import map_debug_info
            exc_msg = (
                "[DIAG retain_graph=True] Failed to run stage backward:\n"
                f"  Stage output: {map_debug_info(stage_output)}\n"
                f"  Output gradient: {map_debug_info(output_grads)}\n"
                f"  Input: {map_debug_info(input_values)}\n"
            )
            raise RuntimeError(exc_msg) from e
        return tuple(grad_inputs)

    _bwd_mod.stage_backward = patched_stage_backward
    setattr(_bwd_mod, _PATCHED_FLAG, True)
    # Also rebind any module that imported it by name. stage.py imports it:
    import torch.distributed.pipelining.stage as _stage_mod
    if hasattr(_stage_mod, "stage_backward"):
        _stage_mod.stage_backward = patched_stage_backward

    logger.warning(
        "AttnRes DIAGNOSTIC: stage_backward now uses retain_graph=True. "
        "If the 'backward graph second time' error disappears, the bug is "
        "intra-call double-traversal; if it persists with a different "
        "error (saved variable modified), it is alias / data corruption."
    )


apply_retain_graph_diag()
