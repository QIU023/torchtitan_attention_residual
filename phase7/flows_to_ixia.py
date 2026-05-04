#!/usr/bin/env python3
"""Convert per-pair flows.csv into an IxNetwork-compatible traffic JSON.

Reads ``flows.csv`` (produced by ``expand_to_flows.py``) and writes
``ixia_config.json`` that an IxNetwork REST client (or IxLoad scenario
loader) can ingest. The schema mirrors IxNetwork's own ``trafficItem``
shape so the consumer can map each item to a stream group with no
further rewriting.

Two emission modes:

* ``--mode flow`` (default): one ``trafficItem`` per logical message
  in flows.csv. Preserves temporal ordering via ``burstStart_us``.
  Output can be very large (millions of items for a 50-step trace);
  IxNetwork generally caps traffic items at ~10k per item-set, so
  use this only on short tier_b/tier_c traces.

* ``--mode aggregated``: bucket flows by (src, dst, axis_guess) and
  emit ONE item per bucket with cumulative byte count and average
  inter-frame gap. Good for steady-state Fabric profiling where the
  ratio of patterns matters more than exact ordering.

Endpoint mapping: ``rank_R`` -> ``card_C/port_P`` round-robin across
4 cards × 2 ports per card (so 8 ranks fill an 8-port test). IPv4
``10.0.0.<R>``, MAC ``02:00:00:00:00:<R:02x>``.

Frame size defaults to ``9000`` (jumbo) which is typical for AI
fabric tests. Override with ``--frame-size``.

Output JSON also includes an ``axisSummary`` block listing total
bytes-on-the-wire per axis so the IXIA test plan can sanity-check
the FSDP/PP/EP split before running.

Usage:
    python phase7/flows_to_ixia.py phase5/runs/<config>/tier_X_trace/ \\
        --frame-size 9000 --mode aggregated
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from math import ceil
from pathlib import Path


def _make_endpoints(n_ranks: int) -> list[dict]:
    eps = []
    for r in range(n_ranks):
        card = (r // 2) + 1
        port = (r % 2) + 1
        eps.append({
            "name": f"rank_{r}",
            "port": f"card_{card}/port_{port}",
            "ipv4": f"10.0.0.{r + 1}",
            "mac": f"02:00:00:00:00:{r:02x}",
        })
    return eps


def _flow_items(flows_path: Path, frame_size: int) -> list[dict]:
    items = []
    with flows_path.open() as f:
        for i, row in enumerate(csv.DictReader(f)):
            bytes_ = int(row["bytes"])
            n_frames = max(1, ceil(bytes_ / frame_size))
            items.append({
                "name": (
                    f"{row['axis_guess']}_op{row['t_us']}"
                    f"_{row['opname']}_c{row['chunk_idx']}_{i}"
                ),
                "type": "L2L3",
                "endpointSet": {
                    "src": [f"rank_{row['src_rank']}"],
                    "dst": [f"rank_{row['dst_rank']}"],
                },
                "frameSize": {"type": "fixed", "value": frame_size},
                "frameCount": n_frames,
                "burstStart_us": int(row["t_us"]),
                "interFrameGap_ns": 12,
                "encapsulation": "ethernetVlan",
                "payload": "incrementing",
                "metadata": {
                    "opname": row["opname"],
                    "axis_guess": row["axis_guess"],
                    "bytes_total": bytes_,
                },
            })
    return items


def _aggregated_items(flows_path: Path, frame_size: int) -> list[dict]:
    """Bucket by (src, dst, axis_guess); one item per bucket with sum
    of bytes and message count.
    """
    buckets: dict[tuple[int, int, str], dict] = defaultdict(
        lambda: {"bytes": 0, "msgs": 0, "ops": set(), "first_t": None,
                 "last_t": None}
    )
    with flows_path.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["src_rank"]), int(row["dst_rank"]),
                   row["axis_guess"])
            b = buckets[key]
            b["bytes"] += int(row["bytes"])
            b["msgs"] += 1
            b["ops"].add(row["opname"])
            t = int(row["t_us"])
            b["first_t"] = t if b["first_t"] is None else min(b["first_t"], t)
            b["last_t"] = t if b["last_t"] is None else max(b["last_t"], t)

    items = []
    for (src, dst, axis), b in sorted(buckets.items()):
        n_frames = max(1, ceil(b["bytes"] / frame_size))
        # synthetic rate: total bytes spread across [first_t, last_t]
        span_us = max(1, (b["last_t"] or 0) - (b["first_t"] or 0))
        items.append({
            "name": f"{axis}_r{src}_to_r{dst}",
            "type": "L2L3",
            "endpointSet": {
                "src": [f"rank_{src}"],
                "dst": [f"rank_{dst}"],
            },
            "frameSize": {"type": "fixed", "value": frame_size},
            "frameCount": n_frames,
            "burstStart_us": b["first_t"] or 0,
            "duration_us": span_us,
            "interFrameGap_ns": 12,
            "encapsulation": "ethernetVlan",
            "payload": "incrementing",
            "metadata": {
                "axis_guess": axis,
                "ops_seen": sorted(b["ops"]),
                "msgs_aggregated": b["msgs"],
                "bytes_total": b["bytes"],
            },
        })
    return items


def _axis_summary(flows_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(
        lambda: {"flows": 0, "bytes": 0, "pairs": set()}
    )
    with flows_path.open() as f:
        for row in csv.DictReader(f):
            a = row["axis_guess"]
            out[a]["flows"] += 1
            out[a]["bytes"] += int(row["bytes"])
            out[a]["pairs"].add(
                (int(row["src_rank"]), int(row["dst_rank"]))
            )
    return {
        a: {
            "flows": v["flows"],
            "bytes_total": v["bytes"],
            "unique_pairs": len(v["pairs"]),
        }
        for a, v in out.items()
    }


def _world_size_from_flows(flows_path: Path) -> int:
    n = 0
    with flows_path.open() as f:
        for row in csv.DictReader(f):
            n = max(n, int(row["src_rank"]) + 1, int(row["dst_rank"]) + 1)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path,
                    help="Trace dir containing flows.csv")
    ap.add_argument("--frame-size", type=int, default=9000,
                    help="Frame size (Jumbo 9000 default; 1500 for "
                         "standard Ethernet)")
    ap.add_argument("--mode", choices=("flow", "aggregated"),
                    default="aggregated",
                    help="flow = one trafficItem per message "
                         "(big output, preserves order); "
                         "aggregated = bucket by (src,dst,axis) "
                         "(small output, steady-state profile).")
    ap.add_argument("--world-size", type=int, default=None)
    args = ap.parse_args()

    flows_path = args.trace_dir / "flows.csv"
    if not flows_path.exists():
        print(f"ERROR: {flows_path} not found. Run "
              f"expand_to_flows.py first.", file=sys.stderr)
        return 1

    n_ranks = args.world_size or _world_size_from_flows(flows_path)
    endpoints = _make_endpoints(n_ranks)
    if args.mode == "flow":
        items = _flow_items(flows_path, args.frame_size)
    else:
        items = _aggregated_items(flows_path, args.frame_size)
    summary = _axis_summary(flows_path)

    out = {
        "schema": "ixnetwork-traffic-config-v1",
        "topology": {"endpoints": endpoints},
        "trafficItems": items,
        "axisSummary": summary,
        "frameSize": args.frame_size,
        "mode": args.mode,
        "source": {
            "trace_dir": str(args.trace_dir),
            "n_traffic_items": len(items),
            "n_endpoints": len(endpoints),
        },
    }
    out_path = args.trace_dir / "ixia_config.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  endpoints={len(endpoints)} traffic_items={len(items)}")
    print("  axis_summary:")
    for a, s in sorted(summary.items()):
        print(f"    {a:8s} flows={s['flows']:>10d} "
              f"bytes={s['bytes_total']:>14d} pairs={s['unique_pairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
