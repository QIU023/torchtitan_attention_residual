#!/usr/bin/env python3
"""Plot Problem A (FSDP baseline + AttnRes N=4) vs Problem B (adapter_pp N=8).

Generates a 4-panel figure:
  - loss curves (3 arms, raw + EMA-smoothed)
  - delta vs baseline (attnres_n4, adapter_n8)
  - throughput (tps per rank)
  - peak memory
"""
from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASE4_RUNS = Path("/root/torchtitan_attention_residual/phase4/runs")
LOGS = OrderedDict([
    ("baseline_fsdp",   PHASE4_RUNS / "kimi_436m_baseline_fsdp_overnight"     / "train.log"),
    ("attnres_fsdp_n4", PHASE4_RUNS / "kimi_436m_block_attn_res_fsdp_overnight" / "train.log"),
    ("adapter_pp_n8",   PHASE4_RUNS / "kimi_pp_adapter_bench" / "adapter_pp"  / "train.log"),
])

ANSI_RE = re.compile(rb"\x1b\[[0-9;]*m")
LINE_RE = re.compile(
    r"step:\s+(\d+)\s+loss:\s+([\d.]+).*?grad_norm:\s+([\d.]+).*?"
    r"memory:\s+([\d.]+)GiB.*?tps:\s+([\d,]+)"
)


def parse(log: Path):
    if not log.exists():
        return None
    raw = log.read_bytes()
    text = ANSI_RE.sub(b"", raw).decode(errors="ignore")
    rows = []
    for m in LINE_RE.finditer(text):
        step = int(m.group(1))
        loss = float(m.group(2))
        grad = float(m.group(3))
        mem = float(m.group(4))
        tps = int(m.group(5).replace(",", ""))
        rows.append((step, loss, grad, mem, tps))
    if not rows:
        return None
    arr = np.array(rows, dtype=np.float64)
    return {
        "step": arr[:, 0],
        "loss": arr[:, 1],
        "grad": arr[:, 2],
        "mem": arr[:, 3],
        "tps": arr[:, 4],
    }


def ema(values: np.ndarray, halflife: int = 50) -> np.ndarray:
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/torchtitan_attention_residual/phase4/runs/kimi_pp_adapter_bench/comparison.png")
    ap.add_argument("--ema-halflife", type=int, default=50)
    args = ap.parse_args()

    data = {name: parse(log) for name, log in LOGS.items()}
    missing = [n for n, d in data.items() if d is None]
    if missing:
        print(f"Missing logs: {missing}")
        return

    colors = {
        "baseline_fsdp":   "#1f77b4",   # blue
        "attnres_fsdp_n4": "#2ca02c",   # green
        "adapter_pp_n8":   "#d62728",   # red
    }
    pretty = {
        "baseline_fsdp":   "Baseline FSDP (dense, N=0)",
        "attnres_fsdp_n4": "AttnRes FSDP (N=4)",
        "adapter_pp_n8":   "Adapter PP (N=8)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    # --- Panel 1: loss curves (raw faded + EMA) ---
    ax = axes[0, 0]
    for name, d in data.items():
        c = colors[name]
        ax.plot(d["step"], d["loss"], color=c, alpha=0.18, linewidth=0.7)
        ax.plot(d["step"], ema(d["loss"], args.ema_halflife),
                color=c, linewidth=1.8, label=pretty[name])
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss (EMA halflife={args.ema_halflife} steps)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(3.5, 8.0)

    # --- Panel 2: delta vs baseline (matched 1000-step grid) ---
    ax = axes[0, 1]
    base = data["baseline_fsdp"]
    base_loss_at = {int(s): l for s, l in zip(base["step"], base["loss"])}
    matched_steps = [s for s in range(1000, 13000, 1000) if s in base_loss_at]
    for name in ("attnres_fsdp_n4", "adapter_pp_n8"):
        d = data[name]
        loss_at = {int(s): l for s, l in zip(d["step"], d["loss"])}
        xs, ys = [], []
        for s in matched_steps:
            if s in loss_at:
                xs.append(s)
                ys.append(loss_at[s] - base_loss_at[s])
        ax.plot(xs, ys, "o-", color=colors[name], label=f"{pretty[name]} − baseline")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Δ loss vs baseline")
    ax.set_title("Loss Δ vs Baseline (1000-step matched samples)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # --- Panel 3: tps (skip warmup) ---
    ax = axes[1, 0]
    for name, d in data.items():
        skip = max(20, len(d["step"]) // 10)
        ax.plot(d["step"][skip:], d["tps"][skip:],
                color=colors[name], alpha=0.4, linewidth=0.6)
        ax.plot(d["step"][skip:], ema(d["tps"][skip:], 100),
                color=colors[name], linewidth=1.8, label=pretty[name])
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/sec/rank")
    ax.set_title("Throughput (tps per rank)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # --- Panel 4: peak memory ---
    ax = axes[1, 1]
    names = list(data.keys())
    peaks = [float(np.max(data[n]["mem"])) for n in names]
    bar_colors = [colors[n] for n in names]
    bars = ax.bar([pretty[n].split(" (")[0] for n in names], peaks, color=bar_colors)
    for b, peak in zip(bars, peaks):
        ax.text(b.get_x() + b.get_width() / 2, peak + 0.3, f"{peak:.2f} GiB",
                ha="center", fontsize=10)
    ax.set_ylabel("Peak memory (GiB)")
    ax.set_title("Peak Reserved Memory per Rank")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(peaks) * 1.18)

    fig.suptitle(
        "Kimi Linear 436M — Problem A (FSDP) vs Problem B (PP+Adapter)\n"
        "12,500 steps × global batch 12 × seq 2048 (≈0.31B tokens)",
        fontsize=13, y=1.02,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
