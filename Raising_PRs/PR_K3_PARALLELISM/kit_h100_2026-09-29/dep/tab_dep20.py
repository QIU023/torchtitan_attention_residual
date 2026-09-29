"""The 20-step DEP numerics: loss and grad norm at steps 1 / 10 / 20 per cell, the steps whose loss and grad
norm equal DEP off's (off_a20), and whether DEP off's loss falls monotonically (the memorisation check)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from steps import steps  # noqa: E402

R = sys.argv[1]
ref = steps(os.path.join(R, "off_a20", "run.log"))
print("| cell | loss 1 / 10 / 20 | grad norm 1 / 10 / 20 | steps equal to DEP off (loss and grad norm) |")
print("|---|---|---|---:|")
for n, label in (("off_a20", "DEP off"), ("off_b20", "DEP off, second run"), ("k25_20", "DEP on, K2.5"),
                 ("bubble20", "DEP on, bubble")):
    r = steps(os.path.join(R, n, "run.log"))
    loss = " / ".join(r[s]["loss"] if s in r else "-" for s in (1, 10, 20))
    gn = " / ".join(r[s]["gn"] if s in r else "-" for s in (1, 10, 20))
    same = sum(1 for s in ref if s in r and (r[s]["loss"], r[s]["gn"]) == (ref[s]["loss"], ref[s]["gn"]))
    print(f"| {label} | {loss} | {gn} | {'reference' if n == 'off_a20' else f'{same} / {len(ref)}'} |")
drops = [s for s in sorted(ref) if s > 1 and s - 1 in ref and float(ref[s]["loss"]) > float(ref[s - 1]["loss"])]
print(f"\nDEP off loss rises at steps: {drops if drops else 'none'}")
