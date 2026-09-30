"""Compare the initial parameter checksums of two probe dumps (dep_probe_local.py): python cmp_params_init.py <a> <b>"""

import glob
import os
import sys

import torch


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    for pa in sorted(glob.glob(os.path.join(a_dir, "params_rank*.pt"))):
        pb = os.path.join(b_dir, os.path.basename(pa))
        a, b = torch.load(pa), torch.load(pb)
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        diff = [k for k in sorted(set(a) & set(b)) if a[k] != b[k]]
        print(f"{os.path.basename(pa)}: {len(a)} / {len(b)} tensors, {len(diff)} differ, only in a {len(only_a)}, only in b {len(only_b)}")
        for k in diff[:12]:
            print(f"   {k}: {a[k]} vs {b[k]}")
        for k in only_a[:4]:
            print(f"   only a: {k}")
        for k in only_b[:4]:
            print(f"   only b: {k}")


if __name__ == "__main__":
    main()
