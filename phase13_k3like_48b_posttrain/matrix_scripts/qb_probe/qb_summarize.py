"""Summarize the qb100 runs: the load table (probe flavors) and the loss table in 4500's format."""
import glob, os, re, statistics, sys
SHORT = "--short" in sys.argv
PREFIX = "/workspace/mx3_qb10_*" if SHORT else "/workspace/mx3_qb100_*"
WIN = ((1, 5), (6, 10), (6, 10)) if SHORT else ((1, 10), (41, 50), (91, 100))
STEPS = (1, 3, 10) if SHORT else (1, 10, 100)
def series(f):
    L, G, P = {}, {}, {}
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(f, errors="ignore").read())
    # the probe's stderr lines can split a trainer log line, so match across them
    for m in re.finditer(r"step:\s*(\d+)(?:.|\n){0,4000}?loss:\s*(-?[0-9.]+)(?:.|\n){0,4000}?grad_norm:\s*([0-9.]+)", text):
        st, loss = int(m.group(1)), float(m.group(2))
        if loss > 0 and st not in L: L[st] = loss; G[st] = float(m.group(3))
    for m in re.finditer(r"LOADPROBE step=(\d+) layer=(\d+) tokens=(\d+) cv=([0-9.]+) max/mean=([0-9.]+) min/mean=([0-9.]+) bmin=(-?[0-9.]+) bmax=(-?[0-9.]+)", text):
        P.setdefault(int(m.group(1)), []).append(tuple(float(x) for x in m.groups()[2:]))
    return L, G, P
runs = {}
for d in sorted(glob.glob(PREFIX), key=os.path.getmtime):
    for f in glob.glob(d + "/*_measure.log"):
        L, G, P = series(f)
        if L: runs[os.path.basename(f).replace("_measure.log", "")] = (L, G, P, os.path.basename(d))
def win(P, a, b, idx):
    vals = [statistics.mean(x[idx] for x in P[s]) for s in range(a, b + 1) if s in P]
    return statistics.mean(vals) if vals else float("nan")
print("### loads (probe flavors): cv over the 32 experts, max/mean, min/mean, averaged over the MoE layers and the step window; bias range at step 100")
print(f"| config | hook | cv {WIN[0][0]}-{WIN[0][1]} | cv {WIN[1][0]}-{WIN[1][1]} | cv {WIN[2][0]}-{WIN[2][1]} | max/mean {WIN[2][0]}-{WIN[2][1]} | min/mean {WIN[2][0]}-{WIN[2][1]} | bias range at {STEPS[-1]} | steps |")
print("|---|---|---|---|---|---|---|---|---|")
for nm, hook in (("dp1_ctrl", "sign-step (main)"), ("dp1_qb", "quantile balancing"), ("dp2_ctrl", "sign-step (main)"), ("dp2_qb", "quantile balancing"), ("dp2_ep2_ctrl", "sign-step (main)"), ("dp2_ep2_qb", "quantile balancing"), ("dp8_ep8_ctrl", "sign-step (main)"), ("dp8_ep8_qb", "quantile balancing"), ("dp1_frozen_ctrl", "sign-step, lr 0"), ("dp1_frozen_qb", "quantile balancing, lr 0")):
    if nm not in runs: print(f"| {nm} | {hook} | missing |"); continue
    L, G, P, d = runs[nm]; last = max(P) if P else 0
    brange = f"[{min(x[4] for x in P[last]):.3f}, {max(x[5] for x in P[last]):.3f}]" if P else "-"
    cfg = nm.replace("_ctrl", "").replace("_qb", "").replace("_", " x ")
    print(f"| {cfg} | {hook} | {win(P,*WIN[0],1):.2f} | {win(P,*WIN[1],1):.2f} | {win(P,*WIN[2],1):.2f} | {win(P,*WIN[2],2):.1f} | {win(P,*WIN[2],3):.2f} | {brange} | {len(L)} |")
print()
print(f"### loss and grad norm, 4500's format: reference = sign-step of the same parallelism; steps {STEPS}")
print("| config | hook | " + " | ".join(f"step {t} loss (diff)" for t in STEPS) + " | " + " | ".join(f"step {t} grad norm (diff)" for t in STEPS) + " |")
print("|---|---|---|---|---|---|---|---|")
def fmt(v, ref): return f"`{v:.6f}` ({abs(v-ref)/ref*100:.3g}%)" if ref is not None else f"`{v:.6f}`"
pairs = [("this_dp1", "dp1", "sign-step, this commit (reference)", None), ("parent_dp1", "dp1", "sign-step, parent commit", "this_dp1"), ("dp1_ctrl", "dp1", "sign-step + load probe", "this_dp1"), ("dp1_qb", "dp1", "quantile balancing", "this_dp1"),
         ("dp2_ctrl", "dp2", "sign-step (reference)", None), ("dp2_qb", "dp2", "quantile balancing", "dp2_ctrl"), ("dp2_ep2_ctrl", "dp2 x ep2", "sign-step (reference)", None), ("dp2_ep2_qb", "dp2 x ep2", "quantile balancing", "dp2_ep2_ctrl"),
         ("dp8_ep8_ctrl", "dp8 x ep8", "sign-step (reference)", None), ("dp8_ep8_qb", "dp8 x ep8", "quantile balancing", "dp8_ep8_ctrl"), ("dp1_frozen_ctrl", "dp1, lr 0", "sign-step (reference)", None), ("dp1_frozen_qb", "dp1, lr 0", "quantile balancing", "dp1_frozen_ctrl")]
for nm, cfg, hook, ref in pairs:
    if nm not in runs: print(f"| {cfg} | {hook} | missing |"); continue
    L, G, P, d = runs[nm]; R = runs.get(ref) if ref else None
    row = [fmt(L[s], R[0][s] if R and s in R[0] else None) for s in STEPS if s in L] + [fmt(G[s], R[1][s] if R and s in R[1] else None) for s in STEPS if s in G]
    print(f"| {cfg} | {hook} | " + " | ".join(row) + " |")
