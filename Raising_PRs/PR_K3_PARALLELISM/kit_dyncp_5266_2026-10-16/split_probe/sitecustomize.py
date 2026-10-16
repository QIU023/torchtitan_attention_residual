"""Local kit only: with SPLIT_PROBE=1, log every dynamic CP split forward (rank, split images, grids) to stderr."""
import os
import sys

if os.environ.get("SPLIT_PROBE") == "1" and "RANK" in os.environ:  # the torchrun parent has no RANK
    from torchtitan.models.kimi_k3.vision_cp.encoder import MoonViTCPEncoder

    _orig = MoonViTCPEncoder._forward_split

    def _logged(self, pixel_values, grids, plan, cp_group):
        sys.stderr.write(f"SPLIT_PROBE rank {os.environ['RANK']} split {list(plan.split)} of grids {grids}\n")
        return _orig(self, pixel_values, grids, plan, cp_group)

    MoonViTCPEncoder._forward_split = _logged
    sys.stderr.write("SPLIT_PROBE installed\n")
