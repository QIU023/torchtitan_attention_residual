"""Summarise the QB x backend cells: routing imbalance against per-rank load."""
import json
import os
import sys

S = "/workspace/results"
cells = [("qb_std", "QB on", "standard"), ("qb_moonep", "QB on", "moonep"),
         ("noqb_std", "QB off", "standard"), ("noqb_moonep", "QB off", "moonep")]
print(f"{'cell':<14} {'QB':<7} {'backend':<9} {'maxvio mean':>12} {'maxvio max':>11} "
      f"{'rank imb mean':>14} {'rank imb max':>13} {'loss':>9}")
rows = []
for tag, qb, be in cells:
    p = os.path.join(S, f"qb_{tag}.json")
    if not os.path.exists(p):
        print(f"{tag:<14} {qb:<7} {be:<9} {'(missing)':>12}")
        continue
    d = json.load(open(p))
    rows.append((tag, qb, be, d))
    print(f"{tag:<14} {qb:<7} {be:<9} {d['maxvio_mean']:>12.3f} {d['maxvio_max']:>11.3f} "
          f"{d['rank_imbalance_mean']:>14.4f} {d['rank_imbalance_max']:>13.4f} "
          f"{d['loss']:>9.5f}")

if len(rows) == 4:
    by = {(r[1], r[2]): r[3] for r in rows}
    print()
    print("factorisation")
    for qb in ("QB on", "QB off"):
        mv = [by[(qb, b)]["maxvio_mean"] for b in ("standard", "moonep")]
        print(f"  {qb:<7} maxvio: standard {mv[0]:.3f}, moonep {mv[1]:.3f}"
              f"   (routing is the router's, so these should agree)")
    for be in ("standard", "moonep"):
        ri = [by[(q, be)]["rank_imbalance_mean"] for q in ("QB on", "QB off")]
        print(f"  {be:<8} rank imbalance: QB on {ri[0]:.4f}, QB off {ri[1]:.4f}")
