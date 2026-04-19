# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare training loss curves from the baseline and AttnRes runs.

Reads TensorBoard event files from both run dirs and produces a 3-panel
figure:

    [0] full loss curves (raw + smoothed)
    [1] post-warmup zoom with common y-limits tight to the two curves
    [2] per-step delta (attn_res - baseline), interpolated onto shared steps

Prints same-step delta at each logged milestone so the diff is obvious even
without opening the image.

Usage:
    python phase2/compare_losses.py \
        --baseline phase2/runs/baseline/tb \
        --attn_res phase2/runs/attn_res/tb \
        --out phase2/runs/comparison.png
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError as e:
    raise SystemExit(
        "tensorboard not installed. Run: pip install tensorboard matplotlib"
    ) from e


@dataclass
class Curve:
    steps: list[int]
    values: list[float]

    @property
    def final(self) -> float | None:
        return self.values[-1] if self.values else None

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.steps, dtype=float), np.asarray(self.values, dtype=float)


TRAIN_LOSS_TAGS = (
    "loss_metrics/global_avg_loss",
    "loss_metrics/loss",
    "loss",
    "train_loss",
    "train/loss",
)


def _first_available_tag(acc: EventAccumulator, tags: Iterable[str]) -> str | None:
    available = set(acc.Tags().get("scalars", []))
    for tag in tags:
        if tag in available:
            return tag
    return None


def _extract(log_dir: str, tags: Iterable[str]) -> Curve:
    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()
    tag = _first_available_tag(acc, tags)
    if tag is None:
        available = acc.Tags().get("scalars", [])
        print(
            f"[compare_losses] WARN: no matching tag in {log_dir}. "
            f"Tried {list(tags)}. Available: {available[:20]}"
        )
        return Curve(steps=[], values=[])
    events = acc.Scalars(tag)
    return Curve(
        steps=[e.step for e in events],
        values=[e.value for e in events],
    )


def _find_tb_dir(path: str) -> str:
    # Accept either the TB event-file dir, the run root (where tb/ lives), or
    # a pattern with one timestamp subdir.
    if os.path.isdir(os.path.join(path, "tb")):
        path = os.path.join(path, "tb")
    # If there is exactly one timestamped subdir, descend into it.
    if os.path.isdir(path):
        entries = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
        if len(entries) == 1:
            return os.path.join(path, entries[0])
    return path


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean that shrinks the window at the edges.

    np.convolve(mode='same') pads with implicit zeros and drags edge values
    toward zero, producing a spurious dip at the tail of the loss curve.
    Here the window is symmetric around i but truncated to available samples.
    """
    if window <= 1 or y.size == 0:
        return y
    half = window // 2
    out = np.empty_like(y, dtype=float)
    for i in range(y.size):
        lo = max(0, i - half)
        hi = min(y.size, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def _aligned_delta(
    b: Curve, a: Curve
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate attn_res onto baseline's step grid (trimmed to overlap)."""
    if not b.values or not a.values:
        return np.array([]), np.array([])
    bs, bv = b.to_arrays()
    as_, av = a.to_arrays()
    lo = max(bs.min(), as_.min())
    hi = min(bs.max(), as_.max())
    mask = (bs >= lo) & (bs <= hi)
    shared_steps = bs[mask]
    av_interp = np.interp(shared_steps, as_, av)
    bv_on_shared = bv[mask]
    return shared_steps, av_interp - bv_on_shared


def _milestones(b: Curve, a: Curve, steps: Iterable[int]) -> list[tuple[int, float, float, float]]:
    """At each target step, pick the nearest logged value from both curves."""
    bs, bv = b.to_arrays()
    as_, av = a.to_arrays()
    out: list[tuple[int, float, float, float]] = []
    for s in steps:
        if bs.size == 0 or as_.size == 0:
            continue
        bi = int(np.argmin(np.abs(bs - s)))
        ai = int(np.argmin(np.abs(as_ - s)))
        # Require both runs to have a sample within 50 steps of the target;
        # otherwise the "same-step" comparison is misleading (e.g. an
        # in-flight run not yet at that step).
        if abs(bs[bi] - s) > 50 or abs(as_[ai] - s) > 50:
            continue
        out.append((s, bv[bi], av[ai], av[ai] - bv[bi]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--attn_res", required=True)
    parser.add_argument("--out", default="comparison.png")
    parser.add_argument(
        "--smooth", type=int, default=20,
        help="Rolling-mean window (in logged samples, each ~10 steps).",
    )
    parser.add_argument(
        "--warmup_end", type=int, default=500,
        help="x-axis lower bound for the zoomed-loss panel and delta panel.",
    )
    args = parser.parse_args()

    baseline_dir = _find_tb_dir(args.baseline)
    attn_res_dir = _find_tb_dir(args.attn_res)

    baseline = _extract(baseline_dir, TRAIN_LOSS_TAGS)
    attn_res = _extract(attn_res_dir, TRAIN_LOSS_TAGS)

    if not baseline.values or not attn_res.values:
        raise SystemExit("[compare_losses] empty curves; check --baseline/--attn_res paths")

    bs, bv = baseline.to_arrays()
    as_, av = attn_res.to_arrays()
    bv_s = _rolling_mean(bv, args.smooth)
    av_s = _rolling_mean(av, args.smooth)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # [0] Full curves: raw faint + smoothed bold.
    axes[0].plot(bs, bv, color="gray", alpha=0.25, linewidth=0.8)
    axes[0].plot(as_, av, color="tab:blue", alpha=0.25, linewidth=0.8)
    axes[0].plot(bs, bv_s, color="gray", label="baseline (smoothed)", linewidth=1.6)
    axes[0].plot(as_, av_s, color="tab:blue", label="AttnRes (smoothed)", linewidth=1.6)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("train loss")
    axes[0].set_title(f"Train loss (full, smoothing window={args.smooth})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # [1] Post-warmup zoom: tight y-limits to the two curves inside the zoom.
    b_mask = bs >= args.warmup_end
    a_mask = as_ >= args.warmup_end
    y_pool = np.concatenate([bv_s[b_mask], av_s[a_mask]])
    if y_pool.size:
        y_pad = 0.02 * (y_pool.max() - y_pool.min() + 1e-9)
        axes[1].set_ylim(y_pool.min() - y_pad, y_pool.max() + y_pad)
    axes[1].plot(bs[b_mask], bv_s[b_mask], color="gray", label="baseline", linewidth=1.6)
    axes[1].plot(as_[a_mask], av_s[a_mask], color="tab:blue", label="AttnRes", linewidth=1.6)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("train loss")
    axes[1].set_title(f"Post-warmup zoom (step >= {args.warmup_end})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # [2] Aligned delta curve.
    shared, delta = _aligned_delta(baseline, attn_res)
    if shared.size:
        delta_s = _rolling_mean(delta, args.smooth)
        mask = shared >= args.warmup_end
        axes[2].axhline(0.0, color="black", linewidth=0.8)
        axes[2].fill_between(
            shared[mask],
            delta_s[mask],
            0.0,
            where=delta_s[mask] < 0,
            color="tab:blue",
            alpha=0.2,
            label="AttnRes better",
        )
        axes[2].fill_between(
            shared[mask],
            delta_s[mask],
            0.0,
            where=delta_s[mask] > 0,
            color="tab:red",
            alpha=0.2,
            label="AttnRes worse",
        )
        axes[2].plot(shared[mask], delta_s[mask], color="tab:purple", linewidth=1.6)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("loss delta (attn_res - baseline)")
    axes[2].set_title("Per-step delta (smoothed)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"[compare_losses] wrote {args.out}")

    # Milestone table: nearest-step same-step deltas.
    ms_steps = [500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
    rows = _milestones(baseline, attn_res, ms_steps)
    if rows:
        print("\n=== Milestone deltas (baseline vs AttnRes at same step) ===")
        print(f"  {'step':>7}  {'baseline':>10}  {'attn_res':>10}  {'delta':>10}")
        for s, bv_s, av_s, d in rows:
            sign = "+" if d >= 0 else "-"
            print(f"  {s:>7}  {bv_s:>10.4f}  {av_s:>10.4f}  {sign}{abs(d):>8.4f}")


if __name__ == "__main__":
    main()
