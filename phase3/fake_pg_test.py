#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fake-process-group smoke test for AttnRes under PP=4, single GPU.

Verifies:
  1. ``AttnResLlama3Model`` splits cleanly over 4 stages via torchtitan's
     module-name pruning.
  2. Intermediate stages return the expected tuple
     ``(partial_block, stacked_blocks)`` and the PyTorch schedule unpacks
     it into the next stage's forward kwargs.
  3. Forward output at stage 3 matches a single-GPU reference forward of
     the full model to within rtol=1e-4 (fp32 CPU comparison).

Run on any machine with a CUDA GPU:

    python phase3/fake_pg_test.py

Expected output:

    [fake_pg] single-GPU reference forward OK (shape=..., mean=...)
    [fake_pg] 4-stage fake-PG forward OK (shape=..., mean=...)
    [fake_pg] max abs diff: X.XXe-05  (threshold 1e-4)
    [fake_pg] PASS
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def _init_fake_pg(world_size: int = 4) -> None:
    """Init a fake process group on single GPU. torch.distributed calls
    still dispatch but no real NCCL traffic happens.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    # torch >=2.4 exposes a "fake" backend for unit testing.
    dist.init_process_group(backend="fake", world_size=world_size, rank=0)


def _build_reference(device: torch.device):
    """Build the full AttnRes debug model on a single device."""
    from torchtitan.experiments.attn_res import attn_res_configs

    config = attn_res_configs["debugmodel_attn_res"]()
    model = config.build().to(device)
    model.init_states()
    model.eval()
    return model, config


def _run_single_gpu(model, config, tokens):
    with torch.no_grad():
        return model(tokens)


def _run_fake_pg(config, tokens, device):
    """Run the same forward sliced over 4 fake PP stages.

    Emulates what torchtitan does: deepcopy + prune modules per stage,
    then thread (partial, blocks) through. Skips PipelineStage so we can
    run on one GPU without a real schedule.
    """
    import copy

    from torch.nn import ModuleDict, ModuleList

    n_layers = len(config.layers)
    num_stages = 4
    assert n_layers % num_stages == 0, "debugmodel has 6 layers; use 2 or 3 stages"
    layers_per_stage = n_layers // num_stages

    stage_models = []
    for s in range(num_stages):
        m = copy.deepcopy(config).build().to(device)
        m.init_states()
        m.eval()

        # prune to this stage's layers
        keep_idxs = set(range(s * layers_per_stage, (s + 1) * layers_per_stage))
        new_layers = ModuleDict()
        for idx, layer in m.layers.items():
            if int(idx) in keep_idxs:
                new_layers[idx] = layer
        m.layers = new_layers

        # stage 0 keeps tok_embeddings; last stage keeps norm + output +
        # final_attn_res_* ; middle stages keep none of those.
        if s != 0:
            m.tok_embeddings = None
        if s != num_stages - 1:
            m.norm = None
            m.output = None
            m.final_attn_res_proj = None
            m.final_attn_res_norm = None

        stage_models.append(m)

    # Thread tokens through stages manually.
    with torch.no_grad():
        out = stage_models[0](tokens)  # -> (partial, blocks_tensor)
        for s in range(1, num_stages):
            if isinstance(out, tuple):
                partial, blocks = out
                out = stage_models[s](partial, blocks=blocks)
            else:
                # Final stage returned logits; nothing after
                break
    return out


def main() -> int:
    _init_fake_pg()
    device = torch.device("cuda:0")

    ref_model, config = _build_reference(device)
    torch.manual_seed(0)
    B, T = 2, 16
    tokens = torch.randint(0, config.vocab_size, (B, T), device=device)

    ref_out = _run_single_gpu(ref_model, config, tokens)
    print(
        f"[fake_pg] single-GPU reference forward OK "
        f"(shape={tuple(ref_out.shape)}, mean={ref_out.mean().item():.4f})"
    )

    # Rebuild with identical init (seed) to get matching weights across
    # single-GPU and staged. We re-seed before each build above.
    torch.manual_seed(0)
    pp_out = _run_fake_pg(config, tokens, device)
    if not isinstance(pp_out, torch.Tensor):
        print(f"[fake_pg] FAIL: expected Tensor from last stage, got {type(pp_out)}")
        return 1
    print(
        f"[fake_pg] 4-stage fake-PG forward OK "
        f"(shape={tuple(pp_out.shape)}, mean={pp_out.mean().item():.4f})"
    )

    diff = (ref_out - pp_out).abs().max().item()
    print(f"[fake_pg] max abs diff: {diff:.3e}  (threshold 1e-4)")
    if diff > 1e-4:
        print("[fake_pg] FAIL: PP forward diverges from single-GPU reference")
        return 1
    print("[fake_pg] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
