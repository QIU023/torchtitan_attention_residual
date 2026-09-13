"""Compare the full-precision loss / grad norm lines (REPR_LOG=1) of two runs, step by step.
Ranks print concurrently, so records are matched anywhere in the text, not per line.
usage: repr_compare.py <log a> <log b>"""
import re, sys

NUM = r"[-+]?\d\.\d+e[-+]\d+"
PAT = re.compile(rf"REPR (\d+) rank(\d+) loss ({NUM}) gn ({NUM})")


def rd(path):
    loss, gn = {}, {}
    for m in PAT.finditer(open(path, errors="ignore").read()):
        step, lv, gv = int(m.group(1)), float(m.group(3)), float(m.group(4))
        if lv > 0:  # non-last pipeline stages carry a sentinel loss
            loss[step] = lv
        gn[step] = gv
    return loss, gn


(la, ga), (lb, gb) = rd(sys.argv[1]), rd(sys.argv[2])
for s in sorted(set(la) & set(lb)):
    dl = (lb[s] - la[s]) / la[s]
    dg = (gb[s] - ga[s]) / ga[s]
    print(f"step {s:3d}  loss {la[s]:.15e} vs {lb[s]:.15e} rel {dl:+.2e}   gn {ga[s]:.15e} vs {gb[s]:.15e} rel {dg:+.2e}")
