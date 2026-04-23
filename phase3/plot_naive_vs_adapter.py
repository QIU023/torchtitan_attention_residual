#!/usr/bin/env python3
"""Plot naive-PP vs adapter-PP loss curves from phase3/runs/*/train.log.

Both runs share: PP=4, V=2, layers_per_stage=2, llama3_175m_attn_res_L16_n8,
local_bs=global_bs=4, seq_len=2048. Adapter run had
TORCHTITAN_ATTNRES_CACHE=1; naive had it unset. Loss is rank-3 (last PP
stage, the only rank that computes loss).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_STEP_RE = re.compile(
    r"step:\s*(\d+)\s+\S*loss:\s*([0-9.eE+-]+)\s+\S*grad_norm:\s*([A-Za-z0-9.eE+-]+)"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a single train.log file. Returns (steps, losses, grads)."""
    steps: list[int] = []
    losses: list[float] = []
    grads: list[float] = []
    with path.open() as f:
        for raw in f:
            line = _ANSI_RE.sub("", raw)
            m = _STEP_RE.search(line)
            if not m:
                continue
            try:
                s = int(m.group(1))
                loss_str = m.group(2)
                grad_str = m.group(3)
                loss = float(loss_str)
                grad = float("inf") if grad_str == "inf" else float(grad_str)
            except ValueError:
                continue
            # rank-3's loss is real; rank-0's is the -1.0 sentinel. Skip -1.
            if loss < 0:
                continue
            steps.append(s)
            losses.append(loss)
            grads.append(grad)
    return np.array(steps), np.array(losses), np.array(grads)


def parse_run_dir(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse all train.log* files in a run dir and merge by step.

    We preserve train.log.60k (pre-resume) separately from train.log
    (post-resume) so extension runs that truncate the live log don't
    lose early-step data. Merge happens by step index — any duplicates
    prefer the later-ordered log (should not happen in practice).
    """
    all_steps: list[int] = []
    all_losses: list[float] = []
    all_grads: list[float] = []
    logs = sorted(run_dir.glob("train.log*"))  # train.log, train.log.60k, ...
    for log in logs:
        s, l, g = parse_log(log)
        all_steps.extend(s.tolist())
        all_losses.extend(l.tolist())
        all_grads.extend(g.tolist())
    # Sort by step and dedupe
    pairs = sorted(zip(all_steps, all_losses, all_grads))
    seen: dict[int, tuple[float, float]] = {}
    for s, l, g in pairs:
        seen[s] = (l, g)
    steps = np.array(sorted(seen.keys()))
    losses = np.array([seen[s][0] for s in steps])
    grads = np.array([seen[s][1] for s in steps])
    return steps, losses, grads


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean using 'valid' mode (no edge padding artifacts).

    Returns a shorter array; callers should align the x-axis by
    slicing the step array to ``[window-1:]`` or via :func:`smooth_with_steps`.
    """
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def smooth_with_steps(
    steps: np.ndarray, y: np.ndarray, window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean aligned to the right edge of the window. Returns
    (steps_truncated, smoothed) both length ``len(y) - window + 1``.
    """
    if window <= 1 or len(y) < window:
        return steps, y
    sm = smooth(y, window)
    # Each smoothed point y_sm[i] corresponds to y[i:i+window]; align to
    # the rightmost step in that window so the curve represents "last
    # window steps' mean at this step".
    return steps[window - 1 :], sm


def window_mean_at(
    steps: np.ndarray, y: np.ndarray, target: int, half_window: int = 25,
) -> tuple[int, float]:
    """Mean of ``y`` over points whose steps lie in ``[target-hw, target+hw]``.
    Returns (nearest_actual_step, mean_value). Falls back to nearest single
    point when the window captures no data.
    """
    lo = np.searchsorted(steps, target - half_window)
    hi = np.searchsorted(steps, target + half_window, side="right")
    if hi <= lo:
        # No points in window: use nearest single
        idx = int(np.clip(np.searchsorted(steps, target), 0, len(steps) - 1))
        return int(steps[idx]), float(y[idx])
    return int(target), float(y[lo:hi].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs_root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    ap.add_argument(
        "--naive", default="pp4_naive_4gpu",
        help="Directory under runs_root containing naive train.log(.60k).",
    )
    ap.add_argument(
        "--adapter", default="pp4_adapter_4gpu",
        help="Directory under runs_root containing adapter train.log.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "naive_vs_adapter_loss.png",
    )
    ap.add_argument("--smooth_window", type=int, default=50,
                    help="Moving-average window for loss smoothing (steps * 10 for log_freq=10).")
    args = ap.parse_args()

    naive_steps, naive_loss, naive_grad = parse_run_dir(args.runs_root / args.naive)
    adapter_steps, adapter_loss, adapter_grad = parse_run_dir(args.runs_root / args.adapter)

    print(f"naive: {len(naive_steps)} points, step {naive_steps[0]}..{naive_steps[-1]}")
    print(f"adapter: {len(adapter_steps)} points, step {adapter_steps[0]}..{adapter_steps[-1]}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: loss curves
    ax = axes[0]
    ax.plot(naive_steps, naive_loss, color="#1f77b4", alpha=0.25, lw=0.5,
            label="naive PP (raw)")
    ns, nsm = smooth_with_steps(naive_steps, naive_loss, args.smooth_window)
    ax.plot(ns, nsm, color="#1f77b4", lw=1.5,
            label=f"naive PP (rolling MA-{args.smooth_window})")
    ax.plot(adapter_steps, adapter_loss, color="#d62728", alpha=0.25, lw=0.5,
            label="adapter PP (raw)")
    a_s, asm = smooth_with_steps(adapter_steps, adapter_loss, args.smooth_window)
    ax.plot(a_s, asm, color="#d62728", lw=1.5,
            label=f"adapter PP (rolling MA-{args.smooth_window})")
    ax.set_ylabel("training loss")
    ax.set_title(
        "Naive PP vs Cross-stage Adapter PP — "
        "Llama3-AttnRes L16_n8, PP=4 V=2, BS=4, 4× RTX 5090 PCIe"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(top=min(12.5, max(naive_loss.max(), adapter_loss.max()) * 1.05))

    # Bottom: grad_norm log-scale
    ax = axes[1]
    finite_naive = np.isfinite(naive_grad)
    finite_adapter = np.isfinite(adapter_grad)
    ax.semilogy(naive_steps[finite_naive], naive_grad[finite_naive],
                color="#1f77b4", alpha=0.35, lw=0.5, label="naive grad_norm")
    ax.semilogy(adapter_steps[finite_adapter], adapter_grad[finite_adapter],
                color="#d62728", alpha=0.35, lw=0.5, label="adapter grad_norm")
    ax.set_xlabel("step")
    ax.set_ylabel("grad_norm (log)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")

    # Print a summary alignment table — use ±25-step window mean
    # (no edge padding artifacts; represents ~500-sample rolling avg).
    print("\n=== Alignment at shared milestones (±250 step mean) ===")
    milestones = [1, 100, 500, 1000, 5000, 10000, 50000, 100000, 150000, 190000, 200000]
    print(f"{'step':>10}  {'naive':>10}  {'adapter':>10}  {'diff':>10}")
    for m in milestones:
        if m > max(naive_steps[-1], adapter_steps[-1]):
            continue
        if m > naive_steps[-1]:
            # adapter only
            _, l_a = window_mean_at(adapter_steps, adapter_loss, m, half_window=250)
            print(f"{m:>10}  {'--':>10}  {l_a:>10.4f}  {'--':>10}")
            continue
        _, l_n = window_mean_at(naive_steps, naive_loss, m, half_window=250)
        _, l_a = window_mean_at(adapter_steps, adapter_loss, m, half_window=250)
        print(f"{m:>10}  {l_n:>10.4f}  {l_a:>10.4f}  {l_a - l_n:>+10.4f}")


if __name__ == "__main__":
    main()
