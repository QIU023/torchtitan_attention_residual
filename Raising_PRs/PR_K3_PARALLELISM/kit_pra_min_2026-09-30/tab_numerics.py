"""PR A numerics rows from the 09-30 H100 logs (and later runs in the same layout): python tab_numerics.py [<results root>]

Per layout: loss / grad norm at steps 1, 10, 50 and 100 for #4656 (without this PR) and PR A, parsed per rank (only
ranks that print a real loss; they must agree at every step), and the count of steps whose loss and grad norm are
identical between the two trees. Extra runs of a layout (e.g. the fresh-cache floor) are given as name=<run dir>.
"""

import os
import re
import sys

LINE = re.compile(r"\[rank(\d+)\].*?step:\s+(\d+)\s+loss:\s+(-?[\d.]+)\s+grad_norm:\s+([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEPS = (1, 10, 50, 100)
LAYOUTS = [
    ("pp4 x vpp2", "pra_h100", "pp4vp2"),
    ("pp4 x vpp4", "pra_h100", "pp4vp4"),
    ("dp2 x pp2 x vpp2, dim 5120", "pra_h100_d5120", "dp2pp2vp2"),
    ("pp2 x vpp2, dim 5120", "pra_h100_d5120", "pp2vp2"),
    ("pp2 x vpp4, dim 5120", "pra_h100_d5120", "pp2vp4"),
]


def parse(path: str) -> dict[int, tuple[str, str]]:
    per_step: dict[int, dict[int, tuple[str, str]]] = {}
    if not os.path.exists(path):
        return {}
    for line in ANSI.sub("", open(path, errors="replace").read()).splitlines():
        m = LINE.search(line)
        if m and float(m[3]) > 0:
            per_step.setdefault(int(m[2]), {}).setdefault(int(m[1]), (m[3], m[4]))
    out = {}
    for step, ranks in per_step.items():
        values = set(ranks.values())
        if len(values) != 1:
            raise SystemExit(f"{path}: step {step} differs across ranks {ranks}")
        out[step] = values.pop()
    return out


def cell(rec: dict[int, tuple[str, str]], step: int) -> str:
    return f"{rec[step][0]} / {rec[step][1]}" if step in rec else "-"


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else os.path.join(
        os.path.dirname(__file__), "..", "kit_h100_2026-09-30", "results")
    extra = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    print("| layout | tree | " + " | ".join(f"step {s}" for s in STEPS) + " | steps identical to without |")
    print("|---|---|" + "---:|" * (len(STEPS) + 1))
    for label, folder, L in LAYOUTS:
        base = parse(os.path.join(root, folder, f"4656_{L}", "run.log"))
        pra = parse(os.path.join(root, folder, f"pra_{L}", "run.log"))
        same = sum(1 for s in base if pra.get(s) == base[s])
        print(f"| {label} | without this PR | " + " | ".join(cell(base, s) for s in STEPS) + " | |")
        print(f"| {label} | this PR | " + " | ".join(cell(pra, s) for s in STEPS) + f" | {same} / {len(base)} |")
        for name, run in extra.items():
            if name.endswith(L):
                rec = parse(os.path.join(run, "run.log"))
                same = sum(1 for s in base if rec.get(s) == base[s])
                print(f"| {label} | {name[: -len(L) - 1]} | " + " | ".join(cell(rec, s) for s in STEPS) + f" | {same} / {len(base)} |")


if __name__ == "__main__":
    main()
