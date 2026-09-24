"""Admit this GPU's capability in the Attention Gym KDA guard (local, never committed)."""
import pathlib
import sys

import torch

capability = torch.cuda.get_device_capability()
for tree in sys.argv[1:]:
    path = pathlib.Path(tree) / "torchtitan/models/kimi_k3/kda.py"
    text = path.read_text()
    if "LOCAL HACK" in text:
        continue
    old = "        if capability not in {(10, 0), (10, 3)}:\n"
    assert text.count(old) == 1, (tree, "guard line not found")
    new = f"        if capability not in {{(10, 0), (10, 3), {capability}}}:  # LOCAL HACK: kit guard lift\n"
    path.write_text(text.replace(old, new))
    print(f"{tree}: KDA guard admits {capability}")
