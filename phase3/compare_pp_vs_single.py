#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare PP vs single-GPU AttnRes training for numerical drift.

Takes 2-3 TensorBoard log dirs (single GPU reference, naive PP, optionally
adapter PP), aligns them on step, and prints max abs diff of the training
loss curve. Flags any early-step divergence that would indicate PP silently
broke AttnRes numerics.

Usage:
    python phase3/compare_pp_vs_single.py \\
        --single phase3/runs/single_reference/tb \\
        --pp phase3/runs/pp8_naive/tb \\
        --pp_cached phase3/runs/pp8_adapter/tb
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:
    print("tensorboard not installed; run: pip install tensorboard", file=sys.stderr)
    sys.exit(1)

LOSS_TAG = "loss_metrics/global_avg_loss"


def _find_tb_dir(path: str) -> str:
    # Accept either the event-file dir or a parent.
    if os.path.isdir(os.path.join(path, "tb")):
        path = os.path.join(path, "tb")
    if os.path.isdir(path):
        entries = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
        if len(entries) == 1:
            return os.path.join(path, entries[0])
    return path


def _extract(path: str) -> tuple[np.ndarray, np.ndarray]:
    acc = EventAccumulator(_find_tb_dir(path), size_guidance={"scalars": 0})
    acc.Reload()
    if LOSS_TAG not in acc.Tags().get("scalars", []):
        raise SystemExit(f"[compare] {LOSS_TAG} not in {path}")
    events = acc.Scalars(LOSS_TAG)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def _align(s1: np.ndarray, v1: np.ndarray, s2: np.ndarray, v2: np.ndarray):
    lo = max(s1.min(), s2.min())
    hi = min(s1.max(), s2.max())
    common = np.intersect1d(s1[(s1 >= lo) & (s1 <= hi)], s2[(s2 >= lo) & (s2 <= hi)])
    idx1 = np.searchsorted(s1, common)
    idx2 = np.searchsorted(s2, common)
    return common, v1[idx1], v2[idx2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True, help="Single-GPU reference run TB dir")
    ap.add_argument("--pp", required=True, help="Naive PP run TB dir")
    ap.add_argument("--pp_cached", help="PP + adapter run TB dir (optional)")
    ap.add_argument("--tol", type=float, default=1e-2, help="Max acceptable loss diff")
    args = ap.parse_args()

    s_single, v_single = _extract(args.single)
    s_pp, v_pp = _extract(args.pp)

    common, v1, v2 = _align(s_single, v_single, s_pp, v_pp)
    diff = np.abs(v1 - v2)
    print(f"[compare] single vs naive PP: aligned {len(common)} steps")
    print(f"          max abs diff = {diff.max():.4f}  "
          f"mean = {diff.mean():.4f}")
    if diff.max() > args.tol:
        print(f"[compare] FAIL: diff exceeds tolerance {args.tol}")
        return 1

    if args.pp_cached:
        s_c, v_c = _extract(args.pp_cached)
        _, v1c, v2c = _align(s_single, v_single, s_c, v_c)
        dc = np.abs(v1c - v2c)
        print(f"[compare] single vs adapter PP: max abs diff = {dc.max():.4f}")
        if dc.max() > args.tol:
            print("[compare] FAIL (adapter): adapter changes numerics")
            return 1
        # also diff the two PP runs directly
        _, va, vb = _align(s_pp, v_pp, s_c, v_c)
        dab = np.abs(va - vb)
        print(f"[compare] naive-PP vs adapter-PP: max abs diff = {dab.max():.4f}")

    print("[compare] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
