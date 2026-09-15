import re
out = "/workspace/tp_h100_out"
def series(f):
    L = {}
    for line in open(f"{out}/{f}.log", errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.]+).*grad_norm: *([0-9.]+)", line)
        if m: L[int(m.group(1))] = (m.group(2), m.group(3))
    return L
def pct(x, r):
    d = (float(x) - float(r)) / float(r) * 100
    return f"{d:+.2g}%" if abs(d) < 1 else f"{d:+.3g}%"
def table(ref, rows, labels):
    R = series(ref)
    print("| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in rows:
        X = series(name)
        if name == ref:
            cells = [f"`{X[s][0]}`" for s in (1, 10, 20)] + [f"`{X[s][1]}`" for s in (1, 10, 20)]
        else:
            allsame = all(X.get(s) == R[s] for s in R)
            cells = []
            for k in (0, 1):
                for s in (1, 10, 20):
                    if allsame or X[s][k] == R[s][k]:
                        cells.append("identical" if allsame else ("same" if k == 0 else "same"))
                    else:
                        cells.append(f"`{X[s][k]}` ({pct(X[s][k], R[s][k])})")
        print(f"| {labels[name]} | " + " | ".join(cells) + " |")
    print()
print("DP1")
table("tp1_parent", ["tp1_parent", "tp1", "tp1_again", "tp2_sp", "tp2_nosp"],
      {"tp1_parent": "tp=1, main", "tp1": "tp=1, this PR", "tp1_again": "tp=1, this PR, fresh inductor cache", "tp2_sp": "tp=2, SP on", "tp2_nosp": "tp=2, SP off"})
print("DP2")
table("dp2", ["dp2", "dp2_ep2", "dp2_tp2", "dp2_ep2_tp2", "dp2_ep2_tp2_nosp"],
      {"dp2": "dp2", "dp2_ep2": "dp2 x ep2 (no TP)", "dp2_tp2": "dp2 x tp2", "dp2_ep2_tp2": "dp2 x ep2 x tp2", "dp2_ep2_tp2_nosp": "dp2 x ep2 x tp2, SP off"})
print("K27")
table("k27_dp2_parent", ["k27_dp2_parent", "k27_dp2"], {"k27_dp2_parent": "main", "k27_dp2": "this PR"})
for a, b in (("tp1_parent", "tp1"), ("tp1_parent", "tp1_again"), ("k27_dp2_parent", "k27_dp2")):
    A, B = series(a), series(b)
    print(a, b, "identical steps", sum(1 for s in A if A[s] == B.get(s)), "of", len(A))
