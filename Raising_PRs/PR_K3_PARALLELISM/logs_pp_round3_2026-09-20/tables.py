"""Markdown tables for the local matrix: python tables.py <out dir> <reference> <cells...>"""
import os
import re
import sys

out, ref, names = sys.argv[1], sys.argv[2], sys.argv[3:]
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


def allsteps(X, Y):
    common = [s for s in Y if s in X]
    same = sum(1 for s in common if X[s] == Y[s])
    return f"{same}/{len(common)}"


hdr = " | ".join(f"step {s}" for s in STEPS)
print(f"| cell | {hdr} | {hdr} (grad norm) | steps identical (loss, norm) | first loss difference |")
print("|---" * (3 + 2 * len(STEPS)) + "|")
for nm in [ref] + names:
    try:
        L, G = series(f"{out}/{nm}.log")
    except FileNotFoundError:
        print(f"| {nm} | missing |")
        continue

    def fmt(X, Y, s):
        if s not in X:
            return "-"
        if nm == ref or s not in Y:
            return f"`{X[s]:.6f}`"
        if X[s] == Y[s]:
            return f"`{X[s]:.6f}` (bitwise)"
        return f"`{X[s]:.6f}` ({(X[s] - Y[s]) / Y[s] * 100:+.3g}%)"

    ident = "-" if nm == ref else f"{allsteps(L, R[0])}, {allsteps(G, R[1])}"
    first = "-" if nm == ref else str(next((s for s in sorted(R[0]) if s in L and L[s] != R[0][s]), "none"))
    print(
        f"| {nm} | "
        + " | ".join(fmt(L, R[0], s) for s in STEPS)
        + " | "
        + " | ".join(fmt(G, R[1], s) for s in STEPS)
        + f" | {ident} | {first} |"
    )
