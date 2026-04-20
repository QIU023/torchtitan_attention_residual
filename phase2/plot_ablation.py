#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot num_blocks ablation: baseline + several AttnRes N variants.

Reads TB events for each run, aligns on step, plots train loss curves on
one axis and per-N delta vs baseline on a second axis. Also prints a
same-step milestone table.

Usage:
    python phase2/plot_ablation.py \
        --baseline phase2/runs/baseline \
        --runs N6=phase2/runs/attn_res \
               N3=phase2/runs/ablation/llama3_175m_attn_res_n3 \
               N12=phase2/runs/ablation/llama3_175m_attn_res_n12 \
        --out phase2/runs/ablation/comparison.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:
    raise SystemExit("pip install tensorboard matplotlib")


LOSS_TAG = "loss_metrics/global_avg_loss"


def _tb_dir(path: str) -> str:
    if os.path.isdir(os.path.join(path, "tb")):
        path = os.path.join(path, "tb")
    if os.path.isdir(path):
        sub = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
        if len(sub) == 1:
            return os.path.join(path, sub[0])
    return path


def _extract(path: str) -> tuple[np.ndarray, np.ndarray]:
    acc = EventAccumulator(_tb_dir(path), size_guidance={"scalars": 0})
    acc.Reload()
    if LOSS_TAG not in acc.Tags().get("scalars", []):
        raise SystemExit(f"{LOSS_TAG} not in {path}")
    events = acc.Scalars(LOSS_TAG)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def _rolling_mean(y: np.ndarray, window: int = 20) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return y
    half = window // 2
    out = np.empty_like(y, dtype=float)
    for i in range(y.size):
        lo = max(0, i - half)
        hi = min(y.size, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def _align_to(
    ref_steps: np.ndarray, ref_vals: np.ndarray, target_steps: np.ndarray
) -> np.ndarray:
    """Linearly interpolate ref_vals onto target_steps."""
    mask = (target_steps >= ref_steps.min()) & (target_steps <= ref_steps.max())
    out = np.full_like(target_steps, np.nan, dtype=float)
    out[mask] = np.interp(target_steps[mask], ref_steps, ref_vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Key=path entries, e.g. N6=phase2/runs/attn_res",
    )
    ap.add_argument("--out", default="ablation.png")
    ap.add_argument("--smooth", type=int, default=20)
    args = ap.parse_args()

    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bs, bv = _extract(args.baseline)
    runs["baseline"] = (bs, bv)
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"--runs entries must be KEY=PATH, got {spec}")
        key, path = spec.split("=", 1)
        runs[key] = _extract(path)

    # --- figure layout ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "baseline": "gray",
        "N3": "#4c78a8",
        "N6": "#54a24b",
        "N12": "#e45756",
        "N2": "#b279a2",
        "N4": "#eeca3b",
    }

    # Panel 0: smoothed loss curves (skip warmup region for readability)
    for name, (s, v) in runs.items():
        v_s = _rolling_mean(v, args.smooth)
        axes[0].plot(s[s >= 500], v_s[s >= 500], color=colors.get(name), label=name, linewidth=1.5)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("train loss (smoothed)")
    axes[0].set_title("Post-warmup loss curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 1: delta vs baseline (for each N variant)
    axes[1].axhline(0, color="black", linewidth=0.8)
    for name, (s, v) in runs.items():
        if name == "baseline":
            continue
        v_s = _rolling_mean(v, args.smooth)
        baseline_interp = _align_to(bs, _rolling_mean(bv, args.smooth), s)
        delta = v_s - baseline_interp
        mask = (~np.isnan(delta)) & (s >= 500)
        axes[1].plot(s[mask], delta[mask], color=colors.get(name), label=name, linewidth=1.5)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss delta (AttnRes - baseline)")
    axes[1].set_title("Per-N delta vs baseline (smoothed)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"[plot_ablation] wrote {args.out}")

    # --- milestone table ---
    milestones = [500, 2500, 5000, 10000, 15000, 20000]
    print("\n=== Same-step milestones (±50 step window) ===")
    header = f"  {'step':>7}  {'baseline':>10}  " + "  ".join(
        f"{name:>10}  {'Δ':>6}" for name in runs if name != "baseline"
    )
    print(header)
    for s in milestones:
        row = f"  {s:>7}  "
        bi = np.argmin(np.abs(bs - s))
        if abs(bs[bi] - s) > 50:
            row += f"{'—':>10}  "
            for name in runs:
                if name == "baseline":
                    continue
                row += f"{'—':>10}  {'—':>6}  "
            print(row.rstrip())
            continue
        row += f"{bv[bi]:>10.4f}  "
        for name, (rs, rv) in runs.items():
            if name == "baseline":
                continue
            ri = np.argmin(np.abs(rs - s))
            if abs(rs[ri] - s) > 50:
                row += f"{'—':>10}  {'—':>6}  "
            else:
                d = rv[ri] - bv[bi]
                sign = "+" if d >= 0 else "-"
                row += f"{rv[ri]:>10.4f}  {sign}{abs(d):>5.4f}  "
        print(row.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
