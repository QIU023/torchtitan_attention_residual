"""The two #4656 body tables from run_4656_h100.sh's results.

Usage: python tab_4656.py <results dir>
"""
import os
import re
import sys

R = sys.argv[1]
PAT = re.compile(
    r"step:\s+(\d+)\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)\s+memory:\s+([0-9.]+)GiB.*?tps:\s+([0-9,]+)"
)


def steps(cell):
    path = os.path.join(R, cell, "run.log")
    if not os.path.exists(path):
        return None, False
    text = open(path, errors="ignore").read()
    oom = "OutOfMemoryError" in text or "CUDA out of memory" in text
    out = {}
    for line in text.splitlines():
        m = PAT.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
        if m:
            out[int(m.group(1))] = (m.group(2), m.group(3), float(m.group(4)), int(m.group(5).replace(",", "")))
    return out, oom


def equal(a, b):
    return sum(1 for k in a if k in b and a[k][:2] == b[k][:2])


ref, _ = steps("id_main_none")
print("| tree | activation checkpointing | step 1 loss / grad norm | step 10 loss / grad norm "
      "| steps equal to main (loss and grad norm) | peak memory | tps (steps 6 to 10) |")
print("| --- | --- | --- | --- | ---: | ---: | ---: |")
rows = [("main", "none", "id_main_none"), ("main, fresh inductor cache", "none", "id_main_none_fresh"),
        ("this PR", "none", "id_pr_none"), ("main", "selective (the flavor default)", "id_main_selective"),
        ("this PR", "selective (the flavor default)", "id_pr_selective"), ("main", "full", "id_main_full"),
        ("this PR", "full", "id_pr_full")]
for tree, ac, cell in rows:
    s, _ = steps(cell)
    base = ref if cell == "id_main_none" else steps("id_main_" + ("none" if "none" in cell else cell.split("_")[-1]))[0]
    eq = "reference" if cell == "id_main_none" else f"{equal(s, base)} / {len(base)}"
    tps = sum(s[k][3] for k in range(6, 11)) / 5
    print(f"| {tree} | {ac} | `{s[1][0]}` / `{s[1][1]}` | `{s[10][0]}` / `{s[10][1]}` | {eq} "
          f"| {max(v[2] for v in s.values()):.2f} GiB | {tps:.0f} |")

print()
print("| tokens per micro-batch | peak memory main | peak memory this PR | saved | step 3 loss / grad norm, main and this PR | tps main / this PR |")
print("| ---: | ---: | ---: | ---: | --- | ---: |")
for tok in (512, 4096, 8192, 16384, 32768, 65536, 98304, 131072):
    m, m_oom = steps(f"sc_main_{tok}")
    p, p_oom = steps(f"sc_pr_{tok}")
    if m is None and p is None:
        continue
    m3 = m.get(3) if m else None
    p3 = p.get(3) if p else None
    mm = "out of memory" if (m_oom or not m3) else f"{m3[2]:.2f} GiB"
    pm = "out of memory" if (p_oom or not p3) else f"{p3[2]:.2f} GiB"
    if m3 and p3:
        saved = f"{m3[2] - p3[2]:.2f} GiB ({100 * (m3[2] - p3[2]) / m3[2]:.0f}%)"
        same = all(m.get(k, (0, 0))[:2] == p.get(k, (1, 1))[:2] for k in (1, 2, 3))
        loss = f"`{m3[0]}` / `{m3[1]}`" + ("" if same else " (DIFFERS)")
        tps = f"{m3[3]} / {p3[3]}"
    else:
        saved = "fits" if p3 else "-"
        loss = f"this PR `{p3[0]}` / `{p3[1]}`" if p3 else "-"
        tps = f"no step / {p3[3]}" if p3 else "-"
    print(f"| {tok} | {mm} | {pm} | {saved} | {loss} | {tps} |")
