"""Loss and grad norm of two runs, step by step, as printed (bitwise as far as the log shows them).

Usage: python cmp_steps.py <run.log A> <run.log B>
Under pipeline parallelism only the rank with the last stage prints the loss; the other ranks print -1.
"""

import re
import sys

_STEP = re.compile(r"step:\s*(\d+)\s+loss:\s*([-\d.]+)\s+grad_norm:\s*([-\d.]+)")


def parse(path):
    rows = {}
    for line in open(path, errors="replace"):
        m = _STEP.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
        if m and m.group(2) != "-1.00000":
            rows[int(m.group(1))] = (m.group(2), m.group(3))
    return rows


def main():
    a, b = parse(sys.argv[1]), parse(sys.argv[2])
    steps = sorted(set(a) & set(b))
    for s in steps:
        print(f"step {s}: {a[s][0]} / {a[s][1]}  vs  {b[s][0]} / {b[s][1]}" + ("" if a[s] == b[s] else "  DIFFERENT"))
    same = sum(a[s] == b[s] for s in steps)
    print(f"identical {same}/{len(steps)} steps (A has {len(a)}, B has {len(b)})")


if __name__ == "__main__":
    main()
