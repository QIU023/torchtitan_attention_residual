"""Body-style tables from the matrix logs: python body_tables.py <out dir> <ref cell> <label=cell> ...

Row format follows the PR body of 2026-09-13: raw value, then on the next line "identical" or the
signed change against the reference; the label of a row that matches the reference on every step
gets "(all 100 steps identical)". Steps: 1, 10, 20, 50, 100 for loss and grad norm.
"""

import os
import re
import sys

out, ref = sys.argv[1], sys.argv[2]
rows = [a.split("=", 1) for a in sys.argv[3:]]
STEPS = tuple(int(x) for x in os.environ.get("TABLE_STEPS", "1 10 20 50 100").split())


def series(f):
    L, G = {}, {}
    for line in open(f, errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.-]+).*grad_norm: *([0-9.]+)", line)
        if m and float(m.group(2)) > 0:
            L[int(m.group(1))] = float(m.group(2))
            G[int(m.group(1))] = float(m.group(3))
    return L, G


R = series(f"{out}/{ref}.log")
hdr = " | ".join(f"step {s}" for s in STEPS[1:])
print(f"| cell | loss, step 1 | {hdr} | grad norm, step 1 | {hdr} |")
print("| --- |" + " ---: |" * (2 * len(STEPS)))


def cellfmt(X, Y, s, is_ref):
    if s not in X:
        return "-"
    if is_ref:
        return f"`{X[s]:.6f}`" if X is L_ else f"`{X[s]:.4f}`"
    raw = f"`{X[s]:.6f}`" if X is L_ else f"`{X[s]:.4f}`"
    if s in Y and X[s] == Y[s]:
        return raw + "<br>identical"
    return raw + f"<br>{(X[s] - Y[s]) / Y[s] * 100:+.2f}%"


for label, cell in [(None, ref)] + rows:
    try:
        L_, G_ = series(f"{out}/{cell}.log")
    except FileNotFoundError:
        print(f"| {label or cell} | missing |")
        continue
    is_ref = cell == ref
    if not is_ref:
        same_l = all(L_.get(s) == R[0].get(s) for s in R[0])
        same_g = all(G_.get(s) == R[1].get(s) for s in R[1])
        if same_l and same_g:
            label = f"{label} (all {len(R[0])} steps identical)"
        elif same_l:
            ng = sum(1 for s in R[1] if G_.get(s) != R[1].get(s))
            label = f"{label} (loss identical on all {len(R[0])} steps, grad norm on all but {ng})"
    name = label if label else f"{cell} (reference)"
    cells = [cellfmt(L_, R[0], s, is_ref) for s in STEPS] + [cellfmt(G_, R[1], s, is_ref) for s in STEPS]
    print(f"| {name} | " + " | ".join(cells) + " |")
