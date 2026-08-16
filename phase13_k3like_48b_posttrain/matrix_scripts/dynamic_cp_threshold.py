"""Where does splitting one image across a sub-CP group start to pay?

``dynamic_cp_min_patches`` decides whether an image is worth partitioning within a
sub-CP group, and its default of 256 has never been calibrated -- it was a plausible
number (256 pre-merge patches is a 16x16 grid, i.e. a 224x224 image at patch_size 14,
64 tokens after the 2x2 merge) with no measurement behind it. This is the measurement.

The trade is one gather per layer against the imbalance that partitioning removes. Below
the threshold the image-level round robin balances better, because a small image split
across ranks buys a per-layer collective for very little work. Above it, leaving a large
image whole makes one rank in the sub-group carry it alone.

So the quantity to measure is the ViT stage's own time under the two placements at a
fixed image size, and the crossing point is where they meet:

  partitioned  the image's patches split across the sub-group, gather-KV per layer
  whole        the image assigned to one rank, the others idle on it

Sweeps patch counts around the current default and prints both, so the number in the
config can point at a row of output instead of at an argument.

    torchrun --nproc_per_node=2 dynamic_cp_threshold.py

Reports per-size medians over several iterations after a warmup, because the first
iteration through a triton kernel includes its compile and would put the crossing point
wherever the sweep happened to start.
"""

import statistics
import time

import torch
import torch.distributed as dist

from torchtitan.models.kimi_k3 import config_registry as cr
from torchtitan.models.kimi_k3.moonvit import MoonViT

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)

WARMUP, ITERS = 3, 10
# Around the default of 256, half an order of magnitude either side. The interesting
# region is where a partition's collectives stop being amortized.
SIZES = [64, 128, 192, 256, 384, 512, 1024]


def build_tower():
    # The vision config the MATRIX runs, taken from the flavor rather than from
    # MoonViTConfig's defaults: the defaults are K3's released 447M tower and a threshold
    # calibrated on it would not describe the cells that exercise dynamic CP.
    spec = cr.kimi_k3_debugmodel_report_arch().model_spec.model
    vision = spec.vision_config
    torch.manual_seed(0)
    # MoonViT takes the config directly, the same call the multimodal wrapper makes.
    tower = MoonViT(vision).to(device).to(torch.bfloat16)
    with torch.no_grad():
        tower.init_weights()
    return tower, vision


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


tower, vision = build_tower()
patch = vision.patch_size
merge = vision.merge_kernel_size[0] * vision.merge_kernel_size[1]

if rank == 0:
    print(
        f"patches  whole(ms)  split(ms)  ratio   (cp={world}, patch={patch}, "
        f"merge={merge}, median of {ITERS})",
        flush=True,
    )

for n_patches in SIZES:
    # A square-ish grid with the patch count requested; t=1 so patch_count == h*w.
    # The grid must divide the 2x2 merge on BOTH placements, and the split hands each
    # rank side/world rows -- so the side has to be a multiple of 2*world, not just of 2.
    step = 2 * world
    side = max(step, int(round(n_patches**0.5)))
    side -= side % step
    grid = torch.tensor([[1, side, side]], device=device)
    total = side * side
    # [L, C, patch, patch]: patch_embed takes patches_LCHW, not a flattened row.
    pixels = torch.randn(
        total, vision.in_channels, patch, patch, device=device, dtype=torch.bfloat16
    )

    def whole():
        with torch.no_grad():
            tower(pixels, grid)

    def split():
        # This rank's row band of the same image, which is what a partitioned placement
        # gives it. Rows, not patches, because the partition is row-wise (vit_cp_plan).
        rows = side // world
        lo, hi = rank * rows * side, (rank + 1) * rows * side
        sub = pixels[lo:hi]
        sub_grid = torch.tensor([[1, rows, side]], device=device)
        with torch.no_grad():
            tower(sub, sub_grid)

    t_whole = timed(whole)
    t_split = timed(split)
    if rank == 0:
        print(
            f"{total:7d}  {t_whole * 1e3:9.2f}  {t_split * 1e3:9.2f}  "
            f"{t_whole / max(t_split, 1e-9):5.2f}x",
            flush=True,
        )

if rank == 0:
    print(
        "\nRead it as: ratio > 1 means the whole image costs more than this rank's share "
        "of a partition, so partitioning is buying something. The crossing point is the "
        "smallest patch count where the ratio clears 1 by a margin that covers the "
        "per-layer gather this probe does NOT charge for -- so treat the crossing as a "
        "lower bound on dynamic_cp_min_patches, not as the value itself.",
        flush=True,
    )
dist.destroy_process_group()
