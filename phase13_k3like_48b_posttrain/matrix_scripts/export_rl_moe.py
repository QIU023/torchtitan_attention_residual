"""Re-export the debug RL checkpoint now that the flavor carries routed experts.

The RL flavor was all-dense while Stage 2 was being brought up; that made the
rollout model unable to exercise the path K3 actually quantizes. With the MoE
stack restored, the fake HF directory has to be rebuilt from the same seed.

The config half matters as much as the tensors. This directory's config.json
descends from a copy of the reference config, so any field the downscaled model
does not exercise still holds the RELEASED value -- ``num_experts`` stayed at
896 and ``routed_expert_hidden_size`` at 3584 while every dense and text-only
export worked, because those models have no routed experts to read them. The
export writes the fields it knows from the titan config and then re-reads the
directory through ``hf_export_shape_check`` rather than trusting either side.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/tt_4025/torchtitan")

from torchtitan.models.kimi_k3 import kimi_k3_configs  # noqa: E402
from torchtitan.models.kimi_k3.state_dict_adapter import (  # noqa: E402
    KimiK3StateDictAdapter,
)

MODEL_DIR = Path("/root/models/kimi-k3-debug")

torch.manual_seed(42)
config = kimi_k3_configs["debugmodel_rl"](attn_backend="flex", moe_comm_backend="standard")
model = config.build()
model.init_weights(buffer_device=torch.device("cpu"))

adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
hf_sd = adapter.to_hf(model.state_dict())
hf_sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in hf_sd.items()}

from safetensors.torch import save_file  # noqa: E402

save_file(hf_sd, str(MODEL_DIR / "model.safetensors"), metadata={"format": "pt"})

experts = sum(1 for k in hf_sd if "expert" in k)
print(f"exported {len(hf_sd)} tensors, {experts} of them expert tensors")

moe = next(layer.moe for layer in config.layers if layer.moe is not None)
fields = {
    "first_k_dense_replace": 1,
    "num_experts": moe.num_experts,
    "n_routed_experts": moe.num_experts,
    "routed_expert_hidden_size": moe.routed_experts.inner_experts.dim,
    "moe_intermediate_size": moe.routed_experts.inner_experts.hidden_dim,
}
config_path = MODEL_DIR / "config.json"
cfg = json.loads(config_path.read_text())
for name, value in fields.items():
    before = cfg["text_config"].get(name)
    cfg["text_config"][name] = value
    print(f"config.json {name}: {before} -> {value}")
config_path.write_text(json.dumps(cfg, indent=1))

check = Path(__file__).with_name("hf_export_shape_check.py")
sys.exit(subprocess.call([sys.executable, str(check), str(MODEL_DIR)]))
