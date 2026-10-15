import torch
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.models.common.multimodal import get_vision_positions
from torchtitan.models.kimi_k3 import KIMI_K3_SPECIAL_TOKENS
from k3v_cells import kimi_k3_debugmodel_video

config = kimi_k3_debugmodel_video()
tokenizer = MultiModalTokenizer(
    MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS), tokenizer_path="./tests/assets/tokenizer"
)
loader = config.dataloader.build(
    dp_world_size=1,
    dp_rank=0,
    tokenizer=tokenizer,
    max_context_length=config.training.max_context_length,
    num_tokens_per_microbatch=config.training.num_tokens_per_microbatch_per_dp_rank,
)
media = tokenizer.image_id
print("image_id", tokenizer.image_id, "video_id", tokenizer.video_id)
for step, batch in zip(range(4), loader):
    d = batch.as_input_dict()
    tokens, grid = d["input"], d["grid_thw"]
    print(step, "tokens", tokens.shape[0], "valid", int((~d["padding_mask"]).sum()) if "padding_mask" in d else None,
          "items", grid.shape[0], "t values", sorted(set(grid[:, 0].tolist())), "pixel", tuple(d["pixel_values"].shape),
          "videos key", d.get("pixel_values_videos") is not None)
    n = (grid[:, 1] // 2) * (grid[:, 2] // 2)
    pos = get_vision_positions(tokens, n, media)
    print("   runs", [(s, k) for _, s, k in pos[:6]], "...", len(pos), "runs")
    if step == 0:
        first = pos[0][1]
        print("   text before first run:", repr(tokenizer.decode(tokens[max(0, first - 14):first + n[0] + 16].tolist())))
loader.close()
