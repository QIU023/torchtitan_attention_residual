# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Does the vision tower actually execute, and does it receive gradients?

The multimodal flavor used to inherit the text dataloader, which emits no
patches. ``KimiK3MultimodalModel.forward`` then takes its first branch --

    if patches is None or not sentinel_present:
        return self.language_model(input_ids)

-- so the tower is constructed, sharded and handed to the optimizer while never
running. Every "multimodal" parallelism leg passed while validating nothing
vision-side. A loss curve cannot distinguish this from a working run.

This wraps the real trainer, counts ``encode_images`` calls, and reports how
many vision-tower parameters ended up with a non-zero gradient. Run it exactly
like ``torchtitan.train``:

    torchrun --nproc_per_node=4 vision_liveness_probe.py --module kimi_k3 \
        --config kimi_k3_mini_vl ...
"""

import os
import sys

import torch

import torchtitan.models.kimi_k3.multimodal_model as mm_mod

_STATE = {"encode_calls": 0, "forward_calls": 0, "text_branch": 0, "logged": False}
_MODELS: list[torch.nn.Module] = []


def _install() -> None:
    cls = mm_mod.KimiK3MultimodalModel
    orig_encode, orig_forward = cls.encode_images, cls.forward

    def encode_images(self, *args, **kwargs):
        _STATE["encode_calls"] += 1
        return orig_encode(self, *args, **kwargs)

    def forward(self, input_ids, pixel_values=None, grid_thw=None, **kwargs):
        _STATE["forward_calls"] += 1
        if self not in _MODELS:
            _MODELS.append(self)
        sentinel = bool((input_ids == self.config.vision_token_id).any().item())
        if pixel_values is None or not sentinel:
            _STATE["text_branch"] += 1
        if not _STATE["logged"] and _rank() == 0:
            _STATE["logged"] = True
            shape = "None" if pixel_values is None else tuple(pixel_values.shape)
            print(
                f"[probe] first forward: pixel_values={shape} sentinel_in_ids={sentinel}",
                flush=True,
            )
        return orig_forward(self, input_ids, pixel_values, grid_thw, **kwargs)

    cls.encode_images = encode_images
    cls.forward = forward


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _report() -> None:
    if _rank() != 0:
        return
    live = dead = 0
    for model in _MODELS:
        tower = getattr(model, "vision_tower", None)
        if tower is None:
            continue
        for p in tower.parameters():
            g = p.grad
            if g is None:
                dead += 1
                continue
            g = g.to_local() if hasattr(g, "to_local") else g
            live += 1 if bool(g.abs().sum().item() > 0) else 0
            dead += 0 if bool(g.abs().sum().item() > 0) else 1
    print(
        f"\n[probe] forward calls        : {_STATE['forward_calls']}"
        f"\n[probe] text-only branch     : {_STATE['text_branch']}"
        f"\n[probe] encode_images calls  : {_STATE['encode_calls']}"
        f"\n[probe] tower params w/ grad : {live} (without: {dead})"
        f"\n[probe] verdict              : "
        + (
            "VISION LIVE"
            if _STATE["encode_calls"] > 0 and live > 0
            else "VISION DEAD -- tower never ran or got no gradient"
        ),
        flush=True,
    )


if __name__ == "__main__":
    _install()
    import atexit

    atexit.register(_report)
    from torchtitan.train import main

    sys.exit(main() or 0)
