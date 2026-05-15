#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LOCAL_RANK -> physical-GPU wrapper for running N ranks on M<N GPUs.

Used by the Phase-3 4-GPU launcher to run 8 PP ranks on 4 physical GPUs
(2 processes per GPU). torchrun still spawns 8 processes with
LOCAL_RANK in {0..7}; this wrapper runs in each child process BEFORE
any torch import, and:

1. Sets ``CUDA_VISIBLE_DEVICES = LOCAL_RANK % PHYSICAL_GPUS`` so the
   child sees exactly one physical GPU (exposed as ``cuda:0``).
2. Overwrites ``LOCAL_RANK`` to ``0`` so torchtitan's
   ``trainer.py:torch.device(f"{device_type}:{LOCAL_RANK}")`` maps to
   the single visible GPU. The pre-mapping LOCAL_RANK is preserved in
   ``ORIGINAL_LOCAL_RANK`` for any downstream consumer that wants it.
3. Execs ``torchtitan.train`` as ``__main__`` with the original argv.

Without this remapping, rank 5's ``torch.device('cuda:5')`` would raise
since only cuda:0..3 exist on a 4-GPU box.

PHYSICAL_GPUS defaults to the current ``CUDA_VISIBLE_DEVICES`` count or
``torch.cuda.device_count()``; override by exporting ``PHYSICAL_GPUS=N``.
"""

from __future__ import annotations

import os
import runpy
import sys


def _physical_gpus() -> int:
    """Number of physical GPUs this job is allowed to use. Honors
    ``PHYSICAL_GPUS``, then falls back to the pre-remap
    ``CUDA_VISIBLE_DEVICES`` count, then to ``nvidia-smi``-style detection.
    """
    env_override = os.environ.get("PHYSICAL_GPUS")
    if env_override is not None:
        return int(env_override)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip()])

    # Last-resort fallback: ask pynvml-free by using /proc. If that fails,
    # default to 1 so the mapping becomes a no-op.
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL
        ).decode()
        return sum(1 for line in out.splitlines() if line.strip())
    except Exception:
        return 1


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    n_phys = max(_physical_gpus(), 1)
    gpu_id = local_rank % n_phys

    # Pin this child to a single physical GPU. Must happen before any
    # torch / CUDA import so the cuda driver initializes with the
    # correct device set.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["ORIGINAL_LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = "0"

    # Visible banner on stderr for smoke-run debugging.
    sys.stderr.write(
        f"[rank_to_gpu_wrapper] rank={os.environ.get('RANK', '?')} "
        f"local_rank={local_rank}->0 gpu={gpu_id} "
        f"world_size={os.environ.get('WORLD_SIZE', '?')}\n"
    )
    sys.stderr.flush()

    # argv[0] currently points at this wrapper file; fix so downstream
    # code inspecting argv sees the torchtitan.train entrypoint.
    sys.argv[0] = "-m torchtitan.train"
    runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
