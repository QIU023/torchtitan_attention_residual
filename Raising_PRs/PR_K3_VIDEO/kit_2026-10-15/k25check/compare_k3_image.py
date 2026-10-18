"""Local kit: titan's K3 image path against the released K3 image processor on images that need downscaling."""
import json
import sys

import torch
from PIL import Image

from k3ref.kimi_k3_vision_processing import KimiK3VisionProcessor
from torchtitan.components.data.multimodal.utils.image import process_image, resize_to_navit_patch_grid, vision_to_patches

cfg = json.load(open("/workspace/k3qat_mm_hf/preprocessor_config.json"))["media_proc_cfg"]
proc = KimiK3VisionProcessor(media_proc_cfg=cfg)
for path, size in ((sys.argv[1], None), (sys.argv[1], (4000, 2380)), (sys.argv[2], (3000, 5000))):
    image = Image.open(path).convert("RGB")
    if size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    ref = proc.preprocess([{"type": "image", "image": image}], return_tensors="pt")
    titan = process_image(image, patch_size=14, merge_size=2, resize_fn=resize_to_navit_patch_grid,
                          max_patches=cfg["in_patch_limit"], max_patches_per_side=cfg["patch_limit_on_one_side"])
    patches, grid = vision_to_patches(titan, 14, 1, 2, patch_order="raster")
    diff = (patches.float() - ref["pixel_values"].flatten(1).float()).abs()
    print(f"image {image.size}: grid titan {grid.tolist()} k3 {ref['grid_thws'][0].tolist()}, "
          f"pixels max diff {diff.max().item():.3e}, over 1e-6: {(diff > 1e-6).sum().item()}/{diff.numel()}")
