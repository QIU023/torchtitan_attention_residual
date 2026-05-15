#!/usr/bin/env python3
"""Expand NCCL collective summary CSV into per-pair P2P flows for Fabric.

Reads ``collective_summary.csv`` (produced by ``extract_collectives.py``)
and decomposes each collective into the per-pair messages a Fabric
(IxNetwork / IxLoad / MoonGen / ns-3) traffic generator needs:

* AllGather (ring):  (N-1) chunks per rank, chunk_size = bytes / N.
                     Pair = neighbor: rank r -> rank (r+1) mod N.
* ReduceScatter:     same shape as AllGather (ring algorithm).
* AllReduce (ring):  2 * (N-1) chunks per rank, chunk_size = bytes / N
                     (= reduce-scatter + allgather phases).
* Broadcast (tree):  (N-1) messages from root to others, full bytes.
* Send / Recv:       already P2P, single message between known ranks.
* AllToAllSingle / AllToAll:  full mesh, bytes / N between every pair.

Each emitted row is one logical message. Multiple rows per collective
encode both the temporal sequence (chunk index) and the per-pair byte
load. Time ordering uses ``(opcount, chunk_idx)`` as a synthetic
microsecond offset so the downstream IXIA emitter can place items in
sequence.

Axis heuristic columns (``axis_guess``) for the user's
"TP stays SHM, FSDP/PP/EP go to Fabric" split:
  * ``pp``      — opname in {Send, Recv}, nranks == world_size
  * ``tp``      — opname == AllReduce, nranks == 2 (small AR pattern)
  * ``fsdp``    — opname in {AllGather, ReduceScatter}, nranks == 2
  * ``ep``      — opname in {AllToAllSingle, AllToAll}, any nranks
  * ``dp``      — opname in {AllReduce, AllGather, ReduceScatter},
                  nranks == world_size
  * ``unknown`` — anything else

The heuristic uses nranks alone — it cannot distinguish TP from FSDP
when both have the same group size on a single host. For correct axis
labels in production, dump comm-to-axis mapping from the trainer
(see phase7_nccl_traffic_catalog/comm_axis_dump.py — TODO).

Rank inference:
  Each NCCL log file is named ``nccl-rank-<host>-<pid>.log``. The CSV
  rows carry (host, pid). We assign rank = sorted index of pid within
  the trace dir (deterministic with torchrun's local rank ordering).

Usage:
    python phase7_nccl_traffic_catalog/expand_to_flows.py phase5_vlm_multimodal_sft/runs/<config>/tier_X_trace/

Emits ``flows.csv`` next to ``collective_summary.csv``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _build_pid_to_rank(csv_path: Path) -> dict[tuple[str, str], int]:
    """Sort (host, pid) pairs and assign rank = index. Single-host
    trace dirs collapse to host == only host so the index is the
    torchrun local rank."""
    pids: set[tuple[str, str]] = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            pids.add((row["host"], row["pid"]))
    return {key: i for i, key in enumerate(sorted(pids, key=lambda p: int(p[1])))}


def _classify(opname: str, nranks: int, world_size: int) -> str:
    op = opname.lower()
    if op in {"send", "recv"}:
        return "pp"
    if op in {"alltoallsingle", "alltoall", "_alltoallsingle"}:
        return "ep"
    if op == "allreduce":
        if nranks == 2:
            return "tp"
        if nranks == world_size:
            return "dp"
        return "unknown"
    if op in {"allgather", "reducescatter", "_allgather_into_tensor",
             "_reduce_scatter_tensor"}:
        if nranks == 2:
            return "fsdp"
        if nranks == world_size:
            return "dp"
        return "unknown"
    return "unknown"


def _expand_row(
    row: dict, src_rank: int, world_size: int,
) -> list[tuple[int, int, int, int, int, str, str]]:
    """Decompose one CSV row into a list of (t_us, src, dst, bytes,
    chunk_idx, opname, axis_guess) tuples.

    The synthetic ``t_us`` is ``opcount * 1000 + chunk_idx`` so flows
    from the same collective appear in chunk order; flows from later
    collectives have larger timestamps. This is *not* a real wallclock
    — it's a sequence number suitable for IXIA's ``startTime`` field
    when the goal is reproducing the COLLECTIVE ORDERING, not real time.
    """
    op = row["opname"]
    nranks = int(row["nranks"])
    bytes_ = int(row["bytes"])
    opcount = int(row["opcount"])
    base_t = opcount * 1000
    axis = _classify(op, nranks, world_size)
    out: list[tuple[int, int, int, int, int, str, str]] = []

    op_l = op.lower()
    if op_l == "send":
        # NCCL Send: src is `src_rank`, dst is `root` field.
        dst = int(row["root"])
        if dst >= 0:
            out.append((base_t, src_rank, dst, bytes_, 0, op, axis))
    elif op_l == "recv":
        # NCCL Recv: dst is src_rank, src is in `root` field.
        src = int(row["root"])
        if src >= 0:
            out.append((base_t, src, src_rank, bytes_, 0, op, axis))
    elif op_l in {"allgather", "_allgather_into_tensor"}:
        # Ring allgather: each rank r sends bytes/N to (r+1) mod N for
        # (N-1) chunks. We emit only flows originating at src_rank to
        # avoid double-counting when the loop walks each rank's CSV.
        chunk = bytes_ // max(nranks, 1)
        dst = (src_rank + 1) % nranks
        for k in range(nranks - 1):
            out.append((base_t + k, src_rank, dst, chunk, k, op, axis))
    elif op_l in {"reducescatter", "_reduce_scatter_tensor"}:
        chunk = bytes_ // max(nranks, 1)
        dst = (src_rank + 1) % nranks
        for k in range(nranks - 1):
            out.append((base_t + k, src_rank, dst, chunk, k, op, axis))
    elif op_l == "allreduce":
        # Ring allreduce = reduce-scatter + allgather, 2*(N-1) chunks.
        chunk = bytes_ // max(nranks, 1)
        dst = (src_rank + 1) % nranks
        for k in range(2 * (nranks - 1)):
            out.append((base_t + k, src_rank, dst, chunk, k, op, axis))
    elif op_l == "broadcast":
        root = int(row["root"])
        if src_rank == root:
            for r in range(nranks):
                if r == root:
                    continue
                out.append((base_t, src_rank, r, bytes_, 0, op, axis))
    elif op_l in {"alltoallsingle", "alltoall", "_alltoallsingle"}:
        chunk = bytes_ // max(nranks, 1)
        for r in range(nranks):
            if r == src_rank:
                continue
            out.append((base_t, src_rank, r, chunk, 0, op, axis))
    # Other / unknown ops: skip silently
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path,
                    help="Trace dir containing collective_summary.csv")
    ap.add_argument("--world-size", type=int, default=None,
                    help="Override world size (default: inferred from "
                         "max nranks observed)")
    ap.add_argument("--keep-axes", default="pp,fsdp,ep,dp",
                    help="Comma list of axis_guess labels to emit "
                         "(default omits 'tp' so TP stays on SHM)")
    args = ap.parse_args()

    csv_path = args.trace_dir / "collective_summary.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 1

    pid_to_rank = _build_pid_to_rank(csv_path)
    if not pid_to_rank:
        print(f"ERROR: no rows in {csv_path}", file=sys.stderr)
        return 1

    keep_axes = {a.strip() for a in args.keep_axes.split(",") if a.strip()}

    # First pass: infer world_size if not given.
    world_size = args.world_size
    if world_size is None:
        max_n = 0
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                max_n = max(max_n, int(row["nranks"]))
        world_size = max_n or len(pid_to_rank)

    out_path = args.trace_dir / "flows.csv"
    n_in = 0
    n_out = 0
    n_dropped_axis = 0
    axis_counts: dict[str, int] = defaultdict(int)
    with csv_path.open() as fin, out_path.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(
            ["t_us", "src_rank", "dst_rank", "bytes",
             "chunk_idx", "opname", "axis_guess"]
        )
        for row in reader:
            n_in += 1
            src_rank = pid_to_rank[(row["host"], row["pid"])]
            for t_us, src, dst, b, k, op, axis in _expand_row(
                row, src_rank, world_size,
            ):
                axis_counts[axis] += 1
                if axis not in keep_axes:
                    n_dropped_axis += 1
                    continue
                writer.writerow([t_us, src, dst, b, k, op, axis])
                n_out += 1

    print(f"in_rows={n_in} out_flows={n_out} dropped_by_axis={n_dropped_axis}")
    print(f"world_size={world_size} keep_axes={sorted(keep_axes)}")
    print("axis breakdown (all flows):")
    for axis, count in sorted(axis_counts.items(), key=lambda x: -x[1]):
        print(f"  {axis:8s} {count}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
