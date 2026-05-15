"""Runtime monkey-patch for the torchtitan PP+V≥2+LBS≥2 backward graph reuse bug.

See additional_found_issues/torchtitan_pp_lbs_backward_INVESTIGATION.md for
the full root-cause investigation. Short version:

    PipelineStage.forward_one_chunk stores `flat_args + flat_kwargs` directly
    into self.fwd_cache[chunk_id] as input_values. Under ScheduleInterleaved1F1B
    with V≥2 and LBS≥2, the activation recv buffers (args_recv_info[chunk_id]
    .buffer) are allocated once and reused across every step. Without cloning,
    input_values keeps a live ref to the recv buffer; the next step's irecv
    overwrites it; this step's backward walks an autograd graph pointing into
    freed/overwritten saved tensors and crashes:

        RuntimeError: Trying to backward through the graph a second time

This module monkey-patches forward_one_chunk to clone tensor inputs before
caching, so input_values gets its own storage and the next step's recv can
no longer alias into the still-active backward graph.

Cost: one activation-sized memcpy per (microbatch, stage). At our shapes
(LBS=14, SEQ=260, D=1168) this is ~5 MB per call, dominated by the P2P
recv it pairs with. Negligible vs the bug being fixed.

Activation: imported by phase5_vlm_multimodal_sft/train_mm.py at module load before the
torchtitan trainer constructs any pipeline schedule. Idempotent — re-import
is a no-op.

Plan: file an RFC + PR upstream at pytorch/pytorch
(torch/distributed/pipelining/stage.py); this experiment-side hotfix is
removed when the upstream fix lands.
"""
from __future__ import annotations

import logging

import torch
from torch.distributed.pipelining.stage import (  # type: ignore
    _PipelineStageBase,
    PipelineStage,
)

logger = logging.getLogger(__name__)


_PATCHED_FLAG_ATTR = "_attnres_pp_backward_hotfix_applied"


def apply_pp_backward_hotfix() -> None:
    """Idempotent: monkey-patch PipelineStage.forward_one_chunk."""
    if getattr(_PipelineStageBase, _PATCHED_FLAG_ATTR, False):
        return

    original = _PipelineStageBase.forward_one_chunk

    def patched_forward_one_chunk(self, fwd_chunk_id, *args, **kwargs):
        # ATTEMPTED hotfix — DOES NOT WORK in pytorch 2.11.
        #
        # Cloning input_values *after* the original forward computes
        # output_tuple breaks PP's gradient-recovery path: gradient is
        # accumulated on the original recv buffer (the autograd leaf
        # used by output_tuple's saved tensors), but PP's
        # get_bwd_send_ops reads .grad from input_values (now a clone
        # whose .grad is always None). Result: error
        # "[N] for chunk 0 has gradients None and is expecting to send
        # gradients to stage M" at step 1 backward.
        #
        # A real fix requires either (a) per-step recv buffer allocation
        # in _prepare_forward_infra so the buffer is never reused across
        # steps, or (b) reading .grad from args_recv_info[id].buffer in
        # the backward send path instead of from input_values. Both are
        # intrusive pytorch-core changes that don't fit a runtime
        # monkey-patch and need an upstream PR.
        #
        # Until that lands upstream, the workaround in this repo is
        # LBS=1 for any PP+V≥2 run (v10, A3 alignment, A2 alignment all
        # pass at LBS=1).
        #
        # Pass through unchanged so the hotfix is a no-op (signature-
        # agnostic via *args/**kwargs).
        return original(self, fwd_chunk_id, *args, **kwargs)

    _PipelineStageBase.forward_one_chunk = patched_forward_one_chunk
    setattr(_PipelineStageBase, _PATCHED_FLAG_ATTR, True)
    logger.info(
        "AttnRes hotfix: patched PipelineStage.forward_one_chunk to "
        "clone input_values, breaking recv-buffer aliasing across "
        "Interleaved1F1B step boundaries (V≥2 + LBS≥2 case). "
        "See additional_found_issues/torchtitan_pp_lbs_backward_INVESTIGATION.md"
    )


# Apply on import.
apply_pp_backward_hotfix()
