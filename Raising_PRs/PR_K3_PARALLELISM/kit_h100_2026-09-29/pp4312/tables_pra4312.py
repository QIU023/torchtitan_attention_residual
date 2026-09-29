"""PR 4312's matrix for PR A: python tables_pra4312.py <out> <reference> <cell> ...
Per cell, main's loss and grad norm at steps 1/10/50/100 against the dp1 reference, main's steps identical to
the reference, and PR A's steps identical to main (loss and grad norm both). dp1 rows have no PR A run."""
import os
import re
import sys

out, ref, names = sys.argv[1], sys.argv[2], sys.argv[3:]
STEPS = (1, 10, 50, 100)


def series(path):
    L, G = {}, {}
    if not os.path.exists(path):
        return L, G
    for line in open(path, errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.-]+).*grad_norm: *([0-9.]+)", line)
        if m and float(m.group(2)) > 0:
            L.setdefault(int(m.group(1)), m.group(2))
            G.setdefault(int(m.group(1)), m.group(3))
    return L, G


def same(X, Y):
    common = [s for s in Y if s in X]
    return sum(1 for s in common if X[s] == Y[s]), len(common)


RL, RG = series(f"{out}/{ref}.log")
print("| cell | loss " + " / ".join(map(str, STEPS)) + " (main) | grad norm " + " / ".join(map(str, STEPS))
      + " (main) | main = reference (loss, grad norm) | PR A = main (loss, grad norm) |")
print("|---|---|---|---|---|")
print(f"| {ref} (reference) | " + " / ".join(RL.get(s, "-") for s in STEPS) + " | "
      + " / ".join(RG.get(s, "-") for s in STEPS) + " | | |")
for nm in names:
    L, G = series(f"{out}/main_{nm}.log")
    pL, pG = series(f"{out}/pra_{nm}.log")
    main_ref = "%d/%d, %d/%d" % (same(L, RL) + same(G, RG))
    if pL:
        a, n = same(pL, L)
        b, m = same(pG, G)
        pra = f"{a}/{n}, {b}/{m}"
    else:
        pra = "-"
    print(f"| {nm} | " + " / ".join(L.get(s, "-") for s in STEPS) + " | " + " / ".join(G.get(s, "-") for s in STEPS)
          + f" | {main_ref} | {pra} |")
