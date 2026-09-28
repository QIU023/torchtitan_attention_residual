"""G1 (DEP) summary from ~/k927/res/dep (rsynced)."""
import collections
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from steps import step_seconds, steps

R = sys.argv[1]


def path(name):
    return os.path.join(R, name, "run.log")


def rc(name):
    p = os.path.join(R, name, "rc")
    return open(p).read().strip() if os.path.exists(p) else "missing"


def dep_lines(name):
    out = collections.Counter()
    if not os.path.exists(path(name)):
        return out
    for raw in open(path(name), errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        m = re.search(r"(DEP (bubble|bubble backward|vision encode):.*)$", line)
        if m:
            out[m.group(1).strip()] += 1
    return out


print("## the committed cell (unseeded, 10 steps)\n")
r = steps(path("cell"))
print(f"rc {rc('cell')}; loss {r[1]['loss'] if r else '-'} -> {r[max(r)]['loss'] if r else '-'}; peak mem {max((v['mem'] for v in r.values()), default=0):.2f} GiB")
for line, n in dep_lines("cell").most_common():
    print(f"  {n} x {line}")

print("\n## bubble on / off (seed 42, deterministic, one warm cache)\n")
trajs = {c: steps(path(c)) for c in ("on", "off", "on2", "off2")}
for c, t in trajs.items():
    print(f"{c}: rc {rc(c)}, steps {len(t)}")


def first_diff(a, b):
    for k in sorted(set(a) & set(b)):
        if (a[k]["loss"], a[k]["gn"]) != (b[k]["loss"], b[k]["gn"]):
            return k
    return None


for a, b in (("on", "on2"), ("off", "off2"), ("on", "off")):
    d = first_diff(trajs[a], trajs[b])
    print(f"{a} vs {b}: " + ("identical on all %d steps" % len(trajs[a]) if d is None else f"first differs at step {d}"))
print("| step | on | off |")
print("|---:|---|---|")
for k in sorted(trajs["on"]):
    o, f = trajs["on"][k], trajs["off"].get(k)
    print(f"| {k} | {o['loss']} / {o['gn']} | {f['loss'] + ' / ' + f['gn'] if f else '-'} |")
p = os.path.join(R, "cmp_grads.txt")
if os.path.exists(p):
    print("\ngradients, bubble on vs off:\n" + open(p).read())

print("## hundred steps, bubble on / off\n")
h = {c: steps(path(c)) for c in ("h100_on", "h100_off")}
for c in h:
    print(f"{c}: rc {rc(c)}, steps {len(h[c])}")
if h["h100_on"] and h["h100_off"]:
    d = first_diff(h["h100_on"], h["h100_off"])
    print(f"first differing step {d}")
    print("| step | on | off |")
    print("|---:|---|---|")
    for k in (1, 2, 3, 10, 20, 50, 100):
        if k in h["h100_on"]:
            print(f"| {k} | {h['h100_on'][k]['loss']} / {h['h100_on'][k]['gn']} | {h['h100_off'][k]['loss']} / {h['h100_off'][k]['gn']} |")
    lo = min(float(v["loss"]) for v in h["h100_off"].values())
    print(f"lowest loss on the off run: {lo}")

print("\n## mixed data, dp_shard 2 x pp2, even DP ranks text only\n")
for c in ("mixed", "mixed_always"):
    t = steps(path(c))
    print(f"{c}: rc {rc(c)}, steps completed {len(t)}")

print("\n## step time (no determinism, one warm cache, 30 steps; mean over steps 11-30)\n")
print("| config | rc | s/step (log timestamps) | tps mean | peak mem (loss rank) | vs dep_off |")
print("|---|---|---:|---:|---:|---:|")
base = None
for cfg in ("dep_off", "dep_bubble_off", "dep_prefetch", "dep_bubble_on"):
    t = steps(path(f"time_{cfg}"))
    s = step_seconds(t, 11, 30) if t else None
    tps = [t[k]["tps"] for k in range(11, 31) if k in t]
    if cfg == "dep_off":
        base = s
    rel = f"{(s / base - 1) * 100:+.1f}%" if s and base else ""
    print(f"| {cfg} | {rc('time_' + cfg)} | {s:.4f} | {statistics.mean(tps):.0f} | {max(v['mem'] for v in t.values()):.2f} | {rel} |" if s else f"| {cfg} | {rc('time_' + cfg)} | | | | |")
    for line, n in dep_lines(f"time_{cfg}").most_common(3):
        print(f"|   | {n} x {line} | | | | |")
