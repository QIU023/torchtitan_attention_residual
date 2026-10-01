"""Step-1 gradient hashes of two runs (PPMEM_GRAD_DUMP directories): python cmp_grads.py <dir A> <dir B>

Counts the parameters, over every rank's file, whose gradient has the same dtype, shape and bytes in both runs.
"""

import json
import os
import sys


def main() -> None:
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    total = equal = no_grad = 0
    diff = []
    names = sorted(n for n in os.listdir(a_dir) if n.endswith(".json"))
    for name in names:
        a = json.load(open(os.path.join(a_dir, name)))
        path = os.path.join(b_dir, name)
        b = json.load(open(path)) if os.path.exists(path) else {}
        if set(a) != set(b):
            diff.append((name, "parameter sets differ", sorted(set(a) ^ set(b))[:5]))
        for key, va in a.items():
            vb = b.get(key)
            if va is None and vb is None:
                no_grad += 1
                continue
            total += 1
            if va and vb and (va["dtype"], va["shape"], va["sha256"]) == (vb["dtype"], vb["shape"], vb["sha256"]):
                equal += 1
            else:
                diff.append((name, key, va and va["norm"], vb and vb["norm"]))
    print(f"{equal} / {total} parameters with gradients bitwise equal over {len(names)} ranks ({no_grad} without gradients)")
    for d in diff[:20]:
        print("DIFF", d)


if __name__ == "__main__":
    main()
