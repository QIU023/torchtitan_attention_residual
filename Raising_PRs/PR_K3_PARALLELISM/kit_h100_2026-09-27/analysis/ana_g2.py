"""G2 tables from ~/k927/res/attnres (rsynced): identity, AC-off sweep, the list table, the checkpoint table."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from steps import step_seconds, steps

R = sys.argv[1]


def rec(name):
    p = os.path.join(R, name, "run.log")
    return steps(p) if os.path.exists(p) else {}


def rc(name):
    p = os.path.join(R, name, "rc")
    return open(p).read().strip() if os.path.exists(p) else "missing"


def oom(name):
    p = os.path.join(R, name, "run.log")
    if not os.path.exists(p):
        return False
    return "OutOfMemoryError" in open(p, errors="replace").read() or "CUDA out of memory" in open(p, errors="replace").read()


def traj(r):
    return [(k, r[k]["loss"], r[k]["gn"]) for k in sorted(r)]


print("## identity (debug model, 2048 tokens/step in 512-token micro-batches, 10 steps, one warm cache)\n")
print("| tree | AC | rc | step 1 loss / gn | step 10 loss / gn | steps equal to main | peak memory (max reserved) | step 10 mem | tps steps 6-10 |")
print("|---|---|---|---|---|---:|---:|---:|---:|")
for ac in ("none", "selective", "full"):
    ref = rec(f"id_main_{ac}")
    for t in ("main", "l4656", "c4780"):
        name = f"id_{t}_{ac}"
        r = rec(name)
        if not r:
            print(f"| {t} | {ac} | {rc(name)} | | | | | | |")
            continue
        eq = sum(1 for k in r if k in ref and (r[k]["loss"], r[k]["gn"]) == (ref[k]["loss"], ref[k]["gn"]))
        tps = [r[k]["tps"] for k in range(6, 11) if k in r]
        peak = max(v["mem"] for v in r.values())
        print(f"| {t} | {ac} | {rc(name)} | `{r[1]['loss']}` / `{r[1]['gn']}` | `{r[max(r)]['loss']}` / `{r[max(r)]['gn']}` | {eq} / {len(r)} | {peak:.2f} GiB | {r[max(r)]['mem']:.2f} | {sum(tps) / len(tps):.0f} |")
r = rec("id_main_none_fresh")
if r:
    ref = rec("id_main_none")
    eq = sum(1 for k in r if k in ref and (r[k]["loss"], r[k]["gn"]) == (ref[k]["loss"], ref[k]["gn"]))
    tps = [r[k]["tps"] for k in range(6, 11) if k in r]
    print(f"| main, fresh cache | none | {rc('id_main_none_fresh')} | `{r[1]['loss']}` / `{r[1]['gn']}` | `{r[max(r)]['loss']}` / `{r[max(r)]['gn']}` | {eq} / {len(r)} | {max(v['mem'] for v in r.values()):.2f} GiB | | {sum(tps) / len(tps):.0f} |")


def pair_table(title, rows, trees, key):
    print(f"\n## {title}\n")
    head = " | ".join(f"{t} peak" for t in trees)
    print(f"| config | {head} | saved ({trees[-1]} vs {trees[0]}) | step 3 loss / gn equal | tps {' / '.join(trees)} |")
    print("|---|" + "---:|" * len(trees) + "---:|---|---:|")
    for cfg in rows:
        recs = [rec(key(t, cfg)) for t in trees]
        cells = []
        for t, r in zip(trees, recs):
            name = key(t, cfg)
            if r and 3 in r:
                cells.append(f"{max(v['mem'] for v in r.values()):.2f} GiB")
            elif oom(name):
                cells.append("out of memory")
            else:
                cells.append(rc(name))
        saved = ""
        if all(r and 3 in r for r in recs):
            a, b = max(v["mem"] for v in recs[0].values()), max(v["mem"] for v in recs[-1].values())
            saved = f"{a - b:.2f} GiB ({(a - b) / a * 100:.0f}%)"
        eq = ""
        if all(r and 3 in r for r in recs):
            eq = "yes" if all((recs[0][k]["loss"], recs[0][k]["gn"]) == (recs[-1][k]["loss"], recs[-1][k]["gn"]) for k in recs[0] if k in recs[-1]) else "NO"
        tps = " / ".join(str(r[3]["tps"]) if r and 3 in r else "-" for r in recs)
        print(f"| {cfg} | " + " | ".join(cells) + f" | {saved} | {eq} | {tps} |")


pair_table("AC off, debug model, one micro-batch (3 steps; peak = max reserved over the steps)",
           [str(t) for t in (2048, 4096, 8192, 16384)], ["main", "l4656", "c4780"],
           lambda t, c: f"sw_{t}_{c}")
pair_table("list carrier: main vs #4656, 48 layers at dim 2048",
           [f"{ac}_b{b}_{tok}" for ac in ("selective", "full") for b in (4, 12) for tok in (8192, 16384)],
           ["main", "l4656"], lambda t, c: f"ls_{t}_{c}")
pair_table("checkpoint: #4656 vs #4780, AC off, dim 2048, blocks of 12",
           [f"l{L}_{tok}" for L in (24, 48) for tok in (4096, 8192, 16384)],
           ["l4656", "c4780"], lambda t, c: f"ck_{t}_{c}")
