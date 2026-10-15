"""Process the four synthetic clips one by one at 1792 px (no grain), collate one micro-batch, check runs and grids."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bigvid_cells import _big_video_sample
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.utils.image import resize_to_navit_patch_grid
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.models.common.multimodal import get_vision_positions
from torchtitan.models.kimi_k3 import KIMI_K3_SPECIAL_TOKENS
from torchtitan_recipes.tests.datasets import synthetic_multimodal

import grain.python as grain

tokenizer = MultiModalTokenizer(MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS), tokenizer_path="./tests/assets/tokenizer")
rows = [json.loads(line) for line in open(synthetic_multimodal._VIDEO_ROOT / "data.jsonl")][:4]
collator = MultiModalCollator.Config(patch_size=14, temporal_patch_size=1, spatial_merge_size=2, patch_order="raster", build_mrope_positions=False).build(
    context=DatasetBuildContext(tokenizer=tokenizer, max_context_length=8704, num_tokens_per_microbatch=8704, read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1))
)
for row in rows:
    sample = _big_video_sample(
        row, tokenizer, video_root=synthetic_multimodal._VIDEO_ROOT, side=1792, frames_per_chunk=4,
        patch_size=14, temporal_patch_size=1, spatial_merge_size=2, min_pixels=65536, max_pixels=16777216,
        image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), resize_fn=resize_to_navit_patch_grid,
        max_patches=16384, max_patches_per_side=512,
    )
    batch = collator([sample]).as_input_dict()
    grid = batch["grid_thw"]
    n = (grid[:, 1] // 2) * (grid[:, 2] // 2)
    pos = get_vision_positions(batch["input"], n, tokenizer.image_id)
    valid = int((~batch["padding_mask"]).sum())
    print(row["video_path"], "chunks", [tuple(c.shape) for c in sample["pixel_values"]], "grid", grid.tolist(), "patches", tuple(batch["pixel_values"].shape), "valid", valid, "runs", [(s, k) for _, s, k in pos])
