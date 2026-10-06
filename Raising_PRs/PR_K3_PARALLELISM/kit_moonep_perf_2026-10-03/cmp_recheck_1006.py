"""Tables for the 10-06 recheck: per tree the body's columns, MoonEP against standard EP, the trees against each other."""

import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"step:\s*(\d+)\s+loss:\s*([0-9.]+)\s+grad_norm:\s*([0-9.]+)")
PAIRS = [("standard", "moonep"), ("standard_ep2", "moonep_ep2"), ("standard_hsdp", "moonep_hsdp"), ("moonep", "moonep_full")]


def read(root, name):
    steps = {}
    try:
        for raw in open(f"{root}/{name}/run.log", errors="replace"):
            raw = ANSI.sub("", raw)
            m = LINE.search(raw)
            if m and raw.startswith("[rank0]"):
                steps[int(m.group(1))] = (m.group(2), m.group(3))
    except FileNotFoundError:
        pass
    return steps


def same(a, b):
    keys = sorted(set(a) & set(b))
    return sum(a[k] == b[k] for k in keys), len(keys)


def gap(a, b):
    keys = sorted(set(a) & set(b))
    return max((abs(float(a[k][0]) - float(b[k][0])) / float(b[k][0]) for k in keys), default=float("nan"))


root, old_root = sys.argv[1], sys.argv[2]
cells = ["standard", "moonep", "moonep_full", "standard_ep2", "moonep_ep2", "standard_hsdp", "moonep_hsdp", "standard_b"]
for tree in ("new", "old"):
    print(f"== {tree}: step 1 / 10 / 20 (loss / grad norm)")
    for c in cells:
        s = read(root, f"num_{tree}_{c}")
        cols = [f"{s[k][0]} / {s[k][1]}" if k in s else "-" for k in (1, 10, 20)]
        print(f"  {c:14s} " + "  ".join(f"{x:>18s}" for x in cols) + f"  ({len(s)} steps)")
    for ref, cand in PAIRS:
        a, b = read(root, f"num_{tree}_{cand}"), read(root, f"num_{tree}_{ref}")
        step1 = "bitwise" if 1 in a and 1 in b and a[1] == b[1] else "DIFFERS"
        print(f"  {cand} vs {ref}: step 1 {step1}, identical {same(a, b)[0]}/{same(a, b)[1]}, max rel loss gap {gap(a, b):.2e}")
    a, b = read(root, f"num_{tree}_standard"), read(root, f"num_{tree}_standard_b")
    print(f"  standard twice: identical {same(a, b)[0]}/{same(a, b)[1]}")
print("== new head against old head, same cache")
for c in cells:
    n, d = same(read(root, f"num_new_{c}"), read(root, f"num_old_{c}"))
    print(f"  {c:14s} identical {n}/{d}")
print("== plain main against the new head's standard cell")
n, d = same(read(root, "num_main_standard"), read(root, "num_new_standard"))
print(f"  identical {n}/{d}")
print("== old head today against its 10-05 run (other cache)")
for c in cells:
    n, d = same(read(root, f"num_old_{c}"), read(old_root, f"num_{c}"))
    print(f"  {c:14s} identical {n}/{d}")
