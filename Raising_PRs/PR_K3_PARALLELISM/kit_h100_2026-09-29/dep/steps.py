"""Per-step records from a titan log: step, loss, grad norm, memory GiB, tps, timestamp.

Usage as a module: steps(path, rank=None) -> {step: dict}. Lines from torchrun --tee carry a
[rankN]: prefix; rank=None takes the last rank that logs a positive loss.
"""
import datetime
import re

_LINE = re.compile(
    r"(?:\[rank(?P<rank>\d+)\]:)?\[titan\] (?P<ts>\S+ \S+) .*step:\s+(?P<step>\d+)\s+loss:\s+(?P<loss>-?[0-9.]+)"
    r"\s+grad_norm:\s+(?P<gn>[0-9.]+)\s+memory:\s+(?P<mem>[0-9.]+)GiB.*?tps:\s+(?P<tps>[0-9,]+)"
)


def steps(path, rank=None):
    by_rank = {}
    with open(path, errors="replace") as f:
        for raw in f:
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
            m = _LINE.search(line)
            if not m:
                continue
            r = int(m.group("rank")) if m.group("rank") is not None else 0
            ts = datetime.datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            by_rank.setdefault(r, {})[int(m.group("step"))] = dict(
                loss=m.group("loss"),
                gn=m.group("gn"),
                mem=float(m.group("mem")),
                tps=int(m.group("tps").replace(",", "")),
                ts=ts,
            )
    if not by_rank:
        return {}
    if rank is None:
        good = [r for r, rec in by_rank.items() if any(float(v["loss"]) > 0 for v in rec.values())]
        rank = max(good) if good else max(by_rank)
    return by_rank.get(rank, {})


def mem_by_rank(path, step):
    """Peak memory (GiB) each rank logs at ``step``."""
    out = {}
    with open(path, errors="replace") as f:
        for raw in f:
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
            m = _LINE.search(line)
            if m and int(m.group("step")) == step:
                out[int(m.group("rank") or 0)] = float(m.group("mem"))
    return out


def step_seconds(rec, first, last):
    """Wall seconds per step from the log timestamps, over steps first..last."""
    ks = [k for k in range(first - 1, last + 1) if k in rec]
    if len(ks) < 2:
        return None
    return (rec[ks[-1]]["ts"] - rec[ks[0]]["ts"]).total_seconds() / (ks[-1] - ks[0])
