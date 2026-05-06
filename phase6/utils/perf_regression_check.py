#!/usr/bin/env python3
"""Phase 6 task C4: throughput/MFU regression check.

Locks in the verified-config baseline numbers from the v1-v8 pretrain
chain on 4×RTX 5090 (Blackwell sm_120). Future PRs run a small
50-step smoke and pipe the train.log into this script; the script
asserts that tps and memory utilization haven't regressed by more
than ``--tolerance`` (default 5%) vs the locked baseline.

Designed for a CI step like::

    bash phase5/launch_train.sh STEPS=50 OUT_DIR=/tmp/perf_smoke
    python phase6/perf_regression_check.py /tmp/perf_smoke/train.log \\
        --config fsdp4_lm_only_bs8

The locked baselines below are point-in-time measurements from the
phase 6 overnight pretrain (commit 92f60e9). Update the
``BASELINES`` table when intentionally changing kernel paths or
torchtitan version.

Usage::

    python phase6/perf_regression_check.py <train.log> --config <name> [--tolerance 0.05]

Exits 0 on PASS, 1 on regression, 2 on parsing failure.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    """Locked-in steady-state perf numbers for one verified config."""
    description: str
    tps_per_rank_min: float       # tokens/sec/rank floor
    memory_gib_max: float         # peak per-rank memory ceiling
    mfu_min: float                # MFU floor (%)
    sample_step: int              # step at which the baseline was recorded


# Locked baselines as of phase6 overnight pretrain (2026-04-30/05-01).
# ``tps_per_rank_min`` and ``mfu_min`` are 5% below the observed steady
# state; ``memory_gib_max`` is +0.5 GB above observed.
BASELINES: dict[str, Baseline] = {
    # FSDP=4 PP=1 LM-only (Phase 4 baseline / Arm 1 caption story config).
    "fsdp4_lm_only_bs8": Baseline(
        description="FSDP=4 PP=1, LOCAL_BS=8 GBS=32, LM-only",
        tps_per_rank_min=380.0,        # observed ~400, -5%
        memory_gib_max=12.0,            # observed 11.6 GiB
        mfu_min=0.74,                   # observed ~0.78%, -5%
        sample_step=100,
    ),
    # FSDP=4 PP=1 multimodal at GBS=64 (LOCAL_BS=16) — original v1 config.
    "fsdp4_mm_bs64": Baseline(
        description="FSDP=4 PP=1, LOCAL_BS=16 GBS=64, multimodal",
        tps_per_rank_min=875.0,         # observed 925, -5%
        memory_gib_max=18.5,            # observed 17.6 GiB
        mfu_min=1.73,                   # observed 1.82%, -5%
        sample_step=100,
    ),
    # FSDP=4 PP=1 multimodal at GBS=120 (LOCAL_BS=30) — v8 high-BS config.
    "fsdp4_mm_bs120": Baseline(
        description="FSDP=4 PP=1, LOCAL_BS=30 GBS=120, multimodal (v8 baseline)",
        tps_per_rank_min=1600.0,        # observed 1700, -5%
        memory_gib_max=29.0,            # observed 28.12 GiB
        mfu_min=3.14,                   # observed 3.31%, -5%
        sample_step=100,
    ),
    # FSDP=1 PP=4 V=2 + cache adapter (Arm 2 alignment config).
    "pp4_v2_adapter_bs12": Baseline(
        description="FSDP=1 PP=4 V=2 + cache adapter, GBS=12 LOCAL_BS=1, multimodal",
        tps_per_rank_min=125.0,         # observed 130, -5%
        memory_gib_max=10.0,            # observed 8.78 GiB
        mfu_min=0.24,                   # observed 0.25%, -5%
        sample_step=100,
    ),
    # FSDP=2 PP=2 V=2 + cache adapter (A6 partial).
    "fsdp2_pp2_v2_adapter": Baseline(
        description="FSDP=2 PP=2 V=2 + cache adapter, GBS=12 LOCAL_BS=1, multimodal",
        tps_per_rank_min=46.0,          # observed 48, -5%
        memory_gib_max=11.5,            # observed 10.67 GiB
        mfu_min=0.085,                  # observed 0.09%, -5%
        sample_step=100,
    ),
}


_STEP_LINE_RE = re.compile(
    r"step:\s*(\d+)\s+loss:\s*([0-9.]+)\s+grad_norm:\s*([0-9.]+)\s+memory:\s*([0-9.]+)GiB\([0-9.]+%\)\s+tps:\s*([0-9.,]+)\s+tflops:\s*([0-9.]+)\s+mfu:\s*([0-9.]+)%"
)


def parse_train_log(path: str) -> list[dict]:
    """Extract step lines from a torchtitan train.log. Returns a list of
    dicts with keys ``step``, ``loss``, ``grad_norm``, ``memory_gib``,
    ``tps``, ``tflops``, ``mfu``."""
    rows = []
    try:
        with open(path) as f:
            for line in f:
                # Strip ANSI codes
                cleaned = re.sub(r"\x1b\[[0-9;]*m", "", line)
                cleaned = re.sub(r"\[(0|0;[0-9]+|3[0-9]+|38;2;[0-9;]+)m", "", cleaned)
                m = _STEP_LINE_RE.search(cleaned)
                if not m:
                    continue
                rows.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "grad_norm": float(m.group(3)),
                    "memory_gib": float(m.group(4)),
                    "tps": float(m.group(5).replace(",", "")),
                    "tflops": float(m.group(6)),
                    "mfu": float(m.group(7)),
                })
    except FileNotFoundError:
        print(f"ERROR: log not found: {path}", file=sys.stderr)
        sys.exit(2)
    return rows


def check(rows: list[dict], baseline: Baseline, tolerance: float) -> tuple[bool, list[str]]:
    """Compare steady-state rows against locked numbers.

    Sampling: take all rows with step ≥ baseline.sample_step (skipping
    early warmup / compile-cost rows), then average the 5 closest-to-
    sample_step. If fewer than 5 such rows exist, average whatever we
    have.
    """
    if not rows:
        return False, ["no parseable step lines in log"]
    # Drop warmup steps below the sample window's lower edge.
    steady = [r for r in rows if r["step"] >= baseline.sample_step]
    if not steady:
        # Fallback: log too short, just use the latest 5
        steady = rows[-5:]
    near = sorted(steady, key=lambda r: r["step"])[:5]
    if not near:
        return False, ["no step lines after warmup"]
    avg_tps = sum(r["tps"] for r in near) / len(near)
    avg_mfu = sum(r["mfu"] for r in near) / len(near)
    max_mem = max(r["memory_gib"] for r in near)

    failures = []
    if avg_tps < baseline.tps_per_rank_min * (1 - tolerance):
        failures.append(
            f"tps_per_rank avg over last 5 steps = {avg_tps:.1f}, "
            f"floor = {baseline.tps_per_rank_min:.1f} ± {tolerance*100:.0f}% "
            f"(=> {baseline.tps_per_rank_min*(1-tolerance):.1f}); REGRESSED"
        )
    if avg_mfu < baseline.mfu_min * (1 - tolerance):
        failures.append(
            f"mfu avg = {avg_mfu:.3f}%, floor = {baseline.mfu_min:.3f}% ± {tolerance*100:.0f}%; "
            f"REGRESSED"
        )
    if max_mem > baseline.memory_gib_max * (1 + tolerance):
        failures.append(
            f"peak memory = {max_mem:.2f} GiB, ceiling = {baseline.memory_gib_max:.2f} GiB ± {tolerance*100:.0f}%; "
            f"REGRESSED (memory blowup)"
        )

    return (len(failures) == 0), [
        f"sample window: steps {[r['step'] for r in near]}",
        f"tps avg     : {avg_tps:.1f} / floor {baseline.tps_per_rank_min:.1f}",
        f"mfu avg     : {avg_mfu:.3f}% / floor {baseline.mfu_min:.3f}%",
        f"peak memory : {max_mem:.2f} GiB / ceiling {baseline.memory_gib_max:.2f} GiB",
    ] + failures


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("log", help="Path to torchtitan train.log")
    p.add_argument(
        "--config", required=True, choices=sorted(BASELINES.keys()),
        help="Baseline config name to compare against",
    )
    p.add_argument(
        "--tolerance", type=float, default=0.05,
        help="Allowable deviation fraction (default 0.05 = 5%)",
    )
    args = p.parse_args()

    baseline = BASELINES[args.config]
    rows = parse_train_log(args.log)
    ok, lines = check(rows, baseline, args.tolerance)

    print(f"Config: {args.config}  ({baseline.description})")
    for ln in lines:
        print(f"  {ln}")
    print(f"\nVerdict: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
