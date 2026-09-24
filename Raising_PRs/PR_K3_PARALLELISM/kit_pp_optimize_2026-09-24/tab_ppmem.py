import json, sys, os
root = sys.argv[1]
names = sys.argv[2:]
def load(n):
    d = {}
    for r in range(8):
        p = os.path.join(root, n, "mem", f"rank{r}.json")
        if os.path.exists(p):
            d[r] = json.load(open(p))["records"]
    return d
data = {n: load(n) for n in names}
keys = ["max_allocated_gib", "max_reserved_gib", "allocated_now_gib", "recv_buffers_gib", "alloc_retries"]
for k in keys:
    print(f"\n## {k} (records: init, step1, step2, step3, exit)")
    for n in names:
        for r in range(8):
            recs = data[n].get(r)
            if recs is None:
                print(f"{n:14s} r{r}: missing"); continue
            print(f"{n:14s} r{r}: " + "  ".join(f"{x[k]:7.3f}" if isinstance(x[k], float) else f"{x[k]:7d}" for x in recs))
