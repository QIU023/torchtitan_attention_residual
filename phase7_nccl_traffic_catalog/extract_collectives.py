#!/usr/bin/env python3
"""Extract NCCL collective patterns from a trace dir into a CSV.

Reads ``nccl-rank-*.log`` files (produced under
``NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL``) inside a single
``phase7_nccl_traffic_catalog/traces/<config_id>/<tier_X>/`` directory and emits a
``collective_summary.csv`` describing every collective issued during
the trace window.

NCCL_DEBUG=INFO COLL lines look like::

    NCCL INFO opCount 5 sendbuff 0x... recvbuff 0x... count 4194304
        datatype 7 op 0 root 0 comm 0x... [nranks=8] stream 0x...
        : opName=AllReduce ...

The parser captures one row per (rank, opCount) pair: op name,
element count, datatype, participant rank count, root (for ops where
it matters), and a synthetic "size_bytes" computed from datatype + count.

Usage:
    python phase7_nccl_traffic_catalog/extract_collectives.py phase7_nccl_traffic_catalog/traces/<config_id>/<tier_X>/
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# NCCL datatype id -> bytes per element (NCCL 2.x ABI)
NCCL_DTYPE_BYTES = {
    0: 1,   # ncclInt8 / ncclChar
    1: 1,   # ncclUint8
    2: 4,   # ncclInt32 / ncclInt
    3: 4,   # ncclUint32
    4: 8,   # ncclInt64
    5: 8,   # ncclUint64
    6: 2,   # ncclFloat16 / ncclHalf
    7: 4,   # ncclFloat32 / ncclFloat
    8: 8,   # ncclFloat64 / ncclDouble
    9: 2,   # ncclBfloat16
}

# NCCL INFO collective line layout (NCCL 2.28.x):
#   ...NCCL INFO <OpName>: opCount <N> sendbuff <0x..> recvbuff <0x..>
#       count <N> datatype <int> op <int> root <int> comm <0x..>
#       [nranks=<int>] stream <ptr>
#
# The opName comes first (with trailing colon), then opCount, etc.
_COLL_RE = re.compile(
    r"NCCL INFO (?P<opname>\w+):\s+"
    r"opCount\s+(?P<opcount>\S+)\s+"
    r"sendbuff\s+\S+\s+recvbuff\s+\S+\s+"
    r"count\s+(?P<count>\d+)\s+"
    r"datatype\s+(?P<dtype>\d+)\s+"
    r"op\s+(?P<reduce_op>\d+)\s+"
    r"root\s+(?P<root>-?\d+)"
    r".*?\[nranks=(?P<nranks>\d+)\]"
)

_RANK_RE = re.compile(r"nccl-rank-([^-]+)-(\d+)\.log$")


def parse_one_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    m = _RANK_RE.search(path.name)
    host = m.group(1) if m else "?"
    pid = m.group(2) if m else "?"
    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                if "opCount" not in line:
                    continue
                m = _COLL_RE.search(line)
                if not m:
                    continue
                count = int(m.group("count"))
                dtype = int(m.group("dtype"))
                bytes_per_el = NCCL_DTYPE_BYTES.get(dtype, 0)
                rows.append(
                    {
                        "host": host,
                        "pid": pid,
                        "opcount": m.group("opcount"),
                        "opname": m.group("opname"),
                        "count": count,
                        "dtype": dtype,
                        "bytes": count * bytes_per_el,
                        "nranks": int(m.group("nranks")),
                        "root": int(m.group("root")),
                    }
                )
    except FileNotFoundError:
        pass
    return rows


def size_bucket(b: int) -> str:
    """Bucket size for histogram readability."""
    if b == 0:
        return "0"
    if b < 1024:
        return "<1KB"
    if b < 64 * 1024:
        return "1-64KB"
    if b < 1024 * 1024:
        return "64KB-1MB"
    if b < 16 * 1024 * 1024:
        return "1-16MB"
    if b < 256 * 1024 * 1024:
        return "16-256MB"
    return "256MB+"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "trace_dir",
        type=Path,
        help="A phase7 tier dir containing nccl-rank-*.log",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <trace_dir>/collective_summary.csv)",
    )
    args = ap.parse_args()

    if not args.trace_dir.is_dir():
        print(f"ERROR: {args.trace_dir} is not a directory", file=sys.stderr)
        return 2
    out = args.out or (args.trace_dir / "collective_summary.csv")

    log_files = sorted(args.trace_dir.glob("nccl-rank-*.log"))
    if not log_files:
        print(
            f"WARN: no nccl-rank-*.log under {args.trace_dir}",
            file=sys.stderr,
        )
        return 1

    all_rows: list[dict] = []
    for p in log_files:
        all_rows.extend(parse_one_log(p))

    if not all_rows:
        print(
            f"WARN: no NCCL collective lines parsed from "
            f"{len(log_files)} log files",
            file=sys.stderr,
        )

    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "host", "pid", "opcount", "opname",
                "count", "dtype", "bytes", "size_bucket",
                "nranks", "root",
            ],
        )
        w.writeheader()
        for r in all_rows:
            r["size_bucket"] = size_bucket(r["bytes"])
            w.writerow(r)

    # Summary histogram on stdout
    hist = Counter(
        (r["opname"], size_bucket(r["bytes"]), r["nranks"])
        for r in all_rows
    )
    print(f"Wrote {len(all_rows)} collective rows to {out}")
    print(f"Per-(op, size_bucket, nranks) histogram:")
    for (op, bkt, n), c in sorted(
        hist.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"  {op:20s} {bkt:10s} nranks={n:2d}  count={c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
