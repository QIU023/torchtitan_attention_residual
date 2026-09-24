"""Per-step loss / grad norm of each cell against its reference, plus peak memory and tps.

Usage: tables_attnres.py <out> <ref>:<cell>[,<cell>...] [<ref>:<cell>...]
Reads <out>/<cell>/train.log. Steps shown: 1, 10 and the last (the debug set is memorised
past step 20, so no later step is reported); equality is counted over every step.
"""
import re
import sys

ROW = re.compile(
    r"step:\s+(\d+)\s+loss:\s+([-\d.]+)\s+grad_norm:\s+([-\d.]+)\s+memory:\s+([\d.]+)GiB.*?tps:\s+([\d,]+)"
)


def read(d, cell):
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(f"{d}/{cell}/train.log").read())
    steps = {}
    for m in ROW.finditer(text):
        steps[int(m[1])] = (m[2], m[3], float(m[4]), int(m[5].replace(",", "")))
    rc = (re.findall(r"^rc=(\d+)", text, re.M) or ["?"])[-1]
    ac = re.findall(r"Applied (\w+) activation checkpointing|Applied (RegionAC) to", text)
    ac = next((a or b for a, b in ac), "none")
    overrides = re.findall(r"Applied (\d+) override", text)
    return steps, rc, ac, overrides[-1] if overrides else "0"


def main():
    d = sys.argv[1]
    print("| cell | AC | overrides | rc | step 1 loss / grad norm | step 10 loss / grad norm | last step loss / grad norm | steps equal to ref (loss and grad norm) | peak GiB | tps (steps 6 on) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for pair in sys.argv[2:]:
        ref, cells = pair.split(":")
        ref_steps = read(d, ref)[0]
        for cell in [ref, *cells.split(",")]:
            steps, rc, ac, n = read(d, cell)
            if not steps:
                print(f"| {cell} | {ac} | {n} | {rc} | | | | | | |")
                continue
            common = sorted(set(steps) & set(ref_steps))
            equal = sum(steps[s][:2] == ref_steps[s][:2] for s in common)
            first = steps.get(1, ("", ""))
            ten = steps.get(10, ("", ""))
            last_step = max(steps)
            last = steps[last_step]
            tps = [steps[s][3] for s in steps if s >= 6]
            same = "reference" if cell == ref else f"{equal} / {len(common)}"
            print(
                f"| {cell} | {ac} | {n} | {rc} | `{first[0]}` / `{first[1]}` | `{ten[0]}` / `{ten[1]}` | "
                f"`{last[0]}` / `{last[1]}` (step {last_step}) | {same} | "
                f"{max(v[2] for v in steps.values()):.2f} | {f'{sum(tps) / len(tps):.0f}' if tps else 'n/a'} |"
            )


if __name__ == "__main__":
    main()
