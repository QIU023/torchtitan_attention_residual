"""Group A of the 5060 smokes: loss and grad norm per step, main against #4656 per AC mode, plus zinit."""
import re, sys, os
R = sys.argv[1]
pat = re.compile(r"step:\s+(\d+)\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)\s+memory:\s+([0-9.]+)GiB")
def steps(cell):
    out = {}
    for line in open(os.path.join(R, cell, "run.log"), errors="ignore"):
        m = pat.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
        if m:
            out[int(m.group(1))] = (m.group(2), m.group(3), float(m.group(4)))
    return out
print("| AC | main step 1 loss / grad norm | main step 10 loss / grad norm | #4656 steps equal to main | memory main / #4656 (GiB, max over steps) |")
print("|---|---|---|---:|---|")
for ac in ("none", "selective", "full"):
    a, b = steps(f"id_main_{ac}"), steps(f"id_l4656_{ac}")
    eq = sum(1 for k in a if k in b and a[k][:2] == b[k][:2])
    print(f"| {ac} | {a[1][0]} / {a[1][1]} | {a[10][0]} / {a[10][1]} | {eq} / {len(a)} | {max(v[2] for v in a.values()):.2f} / {max(v[2] for v in b.values()):.2f} |")
z = steps("zinit_none")
print(f"\nzinit (#4881, AC none): {len(z)} steps, step 1 {z[1][0]} / {z[1][1]}, step 10 {z[10][0]} / {z[10][1]}")
