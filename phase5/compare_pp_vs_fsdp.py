#!/usr/bin/env python3
"""Compare PP+adapter vs FSDP loss curves at matched steps.

Loads two TensorBoard event dirs (one for each run), aligns their
``loss_metrics/global_avg_loss`` series by step, and reports the
maximum and median absolute deviation. Phase 3's measured FSDP
seed-vs-seed noise band on Llama3 175M was ~0.13 nats; staying inside
that range counts as "loss alignment".

Usage::

    python phase5/compare_pp_vs_fsdp.py \
        --pp   phase5/runs/arm2_pp4_v2_fresh_adapter/tb \
        --fsdp phase5/runs/mm_full_finetune/tb
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_scalar(tb_dir: Path, tag: str) -> dict[int, float]:
    from tensorboard.backend.event_processing import event_accumulator
    if not tb_dir.is_dir():
        raise FileNotFoundError(tb_dir)
    ea = event_accumulator.EventAccumulator(
        str(tb_dir), size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        raise KeyError(
            f"Tag {tag!r} not in {tb_dir}. Available: {available}"
        )
    series = {ev.step: ev.value for ev in ea.Scalars(tag)}
    return series


def _align(a: dict[int, float], b: dict[int, float]) -> list[tuple[int, float, float]]:
    common = sorted(set(a) & set(b))
    return [(s, a[s], b[s]) for s in common]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pp", required=True, help="TB dir for PP+adapter run")
    p.add_argument("--fsdp", required=True, help="TB dir for FSDP baseline run")
    p.add_argument("--tag", default="loss_metrics/global_avg_loss",
                   help="Scalar tag to align (default: %(default)s)")
    p.add_argument("--noise-band", type=float, default=0.13,
                   help="Phase 3 measured FSDP seed-vs-seed nats (default: 0.13)")
    p.add_argument("--out-csv", default="",
                   help="Optional CSV dump of (step, pp_loss, fsdp_loss, abs_delta).")
    args = p.parse_args()

    pp = _load_scalar(Path(args.pp), args.tag)
    fsdp = _load_scalar(Path(args.fsdp), args.tag)
    if not pp or not fsdp:
        print(f"ERROR: empty series ({len(pp)=}, {len(fsdp)=})", file=sys.stderr)
        sys.exit(2)

    rows = _align(pp, fsdp)
    if not rows:
        print("ERROR: no overlapping steps", file=sys.stderr)
        sys.exit(2)

    deltas = [abs(p - f) for _, p, f in rows]
    deltas_sorted = sorted(deltas)
    median = deltas_sorted[len(deltas_sorted) // 2]
    p95 = deltas_sorted[max(0, int(len(deltas_sorted) * 0.95) - 1)]
    print(f"Aligned steps: {len(rows)}")
    print(f"Step range: [{rows[0][0]}, {rows[-1][0]}]")
    print(f"|Δ| max:    {max(deltas):.4f} nats")
    print(f"|Δ| p95:    {p95:.4f} nats")
    print(f"|Δ| median: {median:.4f} nats")
    print(f"Phase 3 noise band: {args.noise_band:.3f} nats")
    verdict = "PASS" if max(deltas) <= args.noise_band else "FAIL"
    print(f"Verdict: {verdict}")

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("step,pp_loss,fsdp_loss,abs_delta\n")
            for s, p_, f_ in rows:
                f.write(f"{s},{p_:.6f},{f_:.6f},{abs(p_ - f_):.6f}\n")
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
