"""Compare two probe dumps of stage 0's spliced vision features (dep_probe_local.py): python cmp_feats.py <a> <b>"""

import glob
import os
import sys

import torch


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    for pa in sorted(glob.glob(os.path.join(a_dir, "rank*_call*.pt"))):
        pb = os.path.join(b_dir, os.path.basename(pa))
        if not os.path.exists(pb):
            print(f"{os.path.basename(pa)}: missing in {b_dir}")
            continue
        a, b = torch.load(pa), torch.load(pb)
        for key in ("vision_embeds", "embeds"):
            x, y = a[key].float(), b[key].float()
            same = x.shape == y.shape and torch.equal(a[key], b[key])
            diff = (x - y).abs().max().item() if x.shape == y.shape else float("nan")
            print(f"{os.path.basename(pa)} {key}: bitwise {same}, max abs diff {diff:.3e}, shape {tuple(x.shape)}")
        print(f"   positions equal {a['positions'] == b['positions']}")


if __name__ == "__main__":
    main()
