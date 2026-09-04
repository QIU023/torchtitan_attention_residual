"""Compare two step-1 gradient dumps (name norm shape dtype sha1 per line): identical
parameters by sha1, and the distribution of the relative fp32-norm difference."""
import statistics, sys

def load(path):
    d = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 5:
            continue
        name, norm, sha = parts[0], float(parts[1]), parts[-1]
        d[name] = (norm, sha)
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
common = [k for k in a if k in b]
same = sum(1 for k in common if a[k][1] == b[k][1])
rel = []
for k in common:
    na, nb = a[k][0], b[k][0]
    if max(na, nb) > 1e-12:
        rel.append((abs(na - nb) / max(na, nb), k, na))
rel.sort(reverse=True)
q = lambda p: sorted(r[0] for r in rel)[min(len(rel) - 1, int(p * len(rel)))]
print(f"params {len(common)} (a {len(a)}, b {len(b)}); sha1-identical {same}; differing {len(common) - same}")
if rel:
    print(f"relative norm diff: median {statistics.median(r[0] for r in rel):.2e}  p90 {q(0.9):.2e}  max {rel[0][0]:.2e}  worst {rel[0][1]} (norm {rel[0][2]:.2e})")
    for r in rel[:5]:
        print(f"   {r[0]:.2e}  {r[1]}")
