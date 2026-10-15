import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from big_cells import _build, _degrees
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.models.common.multimodal import get_vision_positions
from torchtitan.models.kimi_k3 import KIMI_K3_SPECIAL_TOKENS

config = _build(_degrees(""))
tokenizer = MultiModalTokenizer(MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS), tokenizer_path="./tests/assets/tokenizer")
loader = config.dataloader.build(
    dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
    max_context_length=config.training.max_context_length,
    num_tokens_per_microbatch=config.training.num_tokens_per_microbatch_per_dp_rank,
)
for step, batch in zip(range(3), loader):
    d = batch.as_input_dict()
    grid = d["grid_thw"]
    n = (grid[:, 1] // 2) * (grid[:, 2] // 2)
    pos = get_vision_positions(d["input"], n, tokenizer.image_id)
    valid = int((~d["padding_mask"]).sum()) if "padding_mask" in d else None
    print(step, "tokens", d["input"].shape[0], "valid", valid, "grid", grid.tolist(), "patches", tuple(d["pixel_values"].shape), "runs", [(s, k) for _, s, k in pos])
