"""Compare stage-output and embedding checksums of two probe dumps (dep_probe_local.py, PROBE_STAGES=1):
python cmp_stages.py <a> <b>; prints every record that differs, forward order per rank."""

import glob
import os
import sys

import torch


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    for pa in sorted(glob.glob(os.path.join(a_dir, "stages_rank*.pt"))):
        a, b = torch.load(pa), torch.load(os.path.join(b_dir, os.path.basename(pa)))
        keys = sorted(set(a) | set(b), key=str)
        diff = [k for k in keys if a.get(k) != b.get(k)]
        print(f"{os.path.basename(pa)}: {len(a)} / {len(b)} records, {len(diff)} differ")
        for k in diff[:16]:
            print(f"   {k}:\\n      a {a.get(k)}\\n      b {b.get(k)}")


if __name__ == "__main__":
    main()
