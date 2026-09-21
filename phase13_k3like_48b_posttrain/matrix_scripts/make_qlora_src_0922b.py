"""Source DCP for item 8 on the 09-22 tree: the rl LoRA flavor at its seed-42 init, model-only, flat keys,
written by the engine venv's torch so no cross-version rewrite is needed. Then repacked by
scripts/quantize_lora_dcp.py into the stacked-w13 packed layout."""
import sys

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.models.kimi_k3.config_registry import kimi_k3_rl_lora

dst = sys.argv[1]
torch.manual_seed(42)
spec = kimi_k3_rl_lora().model_spec
model = spec.model.build()
if hasattr(model, "init_weights"):
    model.init_weights(buffer_device=torch.device("cpu"))
else:
    model.init_states()
sd = {k: v.detach() for k, v in model.state_dict().items()}
w13 = [k for k in sd if k.endswith("w13.weight")]
print("keys", len(sd), "lora keys", sum("lora_" in k for k in sd), "w13", w13[:2], tuple(sd[w13[0]].shape) if w13 else None)
dcp.save(sd, checkpoint_id=dst)
print("saved", dst)
