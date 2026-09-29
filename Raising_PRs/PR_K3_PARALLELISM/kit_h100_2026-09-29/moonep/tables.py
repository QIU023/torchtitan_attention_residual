"""Tables from smoke.sh's results: loss and grad norm at steps 1/5/10/20 per numerics cell,
the gap to the standard backend, peak memory per rank, and mean step time over steps 11-30."""

import re
import sys
from pathlib import Path

STEP = re.compile(
    r"\[rank(\d+)\].*step:\s+(\d+)\s+loss:\s+([\d.]+)\s+grad_norm:\s+([\d.]+)\s+"
    r"memory:\s+([\d.]+)GiB.*?tps:\s+([\d,]+)"
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read(run: Path) -> dict:
    steps: dict[int, tuple[float, float]] = {}
    memory: dict[int, float] = {}
    tps: dict[int, list[float]] = {}
    for line in ANSI.sub("", run.read_text(errors="replace")).splitlines():
        m = STEP.search(line)
        if not m:
            continue
        rank, step = int(m[1]), int(m[2])
        steps.setdefault(step, (float(m[3]), float(m[4])))
        memory[rank] = max(memory.get(rank, 0.0), float(m[5]))
        tps.setdefault(step, []).append(float(m[6].replace(",", "")))
    return {"steps": steps, "memory": memory, "tps": tps}


def main(out: Path) -> None:
    cells = {p.name: read(p / "run.log") for p in sorted(out.iterdir()) if (p / "run.log").exists()}
    ref = cells.get("num_standard")
    print("numerics (deterministic, one warm cache; cc12m-test is a debug set, so 20 steps)")
    print("| cell | " + " | ".join(f"step {s} loss / grad norm" for s in (1, 5, 10, 20)) + " | max rel loss gap to standard |")
    print("|---|" + "---:|" * 5)
    for name, c in cells.items():
        if not name.startswith("num_"):
            continue
        row = []
        for s in (1, 5, 10, 20):
            v = c["steps"].get(s)
            row.append("-" if v is None else f"{v[0]:.5f} / {v[1]:.4f}")
        gap = "-"
        if ref and c is not ref:
            common = sorted(set(c["steps"]) & set(ref["steps"]))
            if common:
                gap = f"{max(abs(c['steps'][s][0] - ref['steps'][s][0]) / ref['steps'][s][0] for s in common):.2e}"
        print(f"| {name[4:]} | " + " | ".join(row) + f" | {gap} |")
    print("\npeak memory per rank, GiB (numerics cells)")
    for name, c in cells.items():
        if name.startswith("num_"):
            print(f"{name[4:]}: " + ", ".join(f"r{r} {v:.2f}" for r, v in sorted(c["memory"].items())))
    print("\nmean step throughput over steps 11-30 (timing cells, not deterministic), tokens/s per rank")
    for name, c in cells.items():
        if name.startswith("time_"):
            vals = [sum(v) / len(v) for s, v in c["tps"].items() if 11 <= s <= 30]
            print(f"{name[5:]}: {sum(vals) / len(vals):.0f} over {len(vals)} steps" if vals else f"{name[5:]}: no steps")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
