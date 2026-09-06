"""Export the NEW tree's 12-layer debug model (run-worktree flavor `rl`) as the fake HF directory the
verl cells load: /root/models/kimi-k3-debug-nt. Weights from seed 42 through the tree's own state-dict
adapter; config.json is the 09-02 export's, with every field the new shape changes rewritten from the
titan config (MLA 64/32/64, kv_lora_rank 256, KDA head_dim 128 on 16 heads, the 3 KDA : 1 MLA layer
lists for 12 layers, the MoE fields), then re-read through hf_export_shape_check.

Run with /venv/main and PYTHONPATH=/tmp/attn_gym_up:/tmp/wt_k3int.
"""
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from torchtitan.models.kimi_k3 import kimi_k3_configs, kimi_k3_full_attention_layers
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/models/kimi-k3-debug-nt")
FLAVOR = sys.argv[2] if len(sys.argv) > 2 else "rl"

torch.manual_seed(42)
get_config, _ = kimi_k3_configs[FLAVOR]
config = get_config(attn_backend="flex", moe_comm_backend="standard")
model = config.build()
model.init_weights(buffer_device=torch.device("cpu"))

hf_sd = KimiK3StateDictAdapter(config, hf_assets_path=None).to_hf(model.state_dict())
hf_sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in hf_sd.items()}
save_file(hf_sd, str(MODEL_DIR / "model.safetensors"), metadata={"format": "pt"})
print(f"exported {len(hf_sd)} tensors, {sum(1 for k in hf_sd if 'expert' in k)} of them expert tensors")

layers = config.layers
n = len(layers)
attn = layers[-1].attention if layers[-1].attention is not None else next(l.attention for l in layers if l.attention is not None)
kda = next(l.delta_attention for l in layers if l.delta_attention is not None)
moe = next(l.moe for l in layers if l.moe is not None)
full = sorted(i + 1 for i in kimi_k3_full_attention_layers(n))
fields = {
    "num_hidden_layers": n,
    "hidden_size": config.dim if hasattr(config, "dim") else 1024,
    "num_attention_heads": attn.n_heads,
    "num_key_value_heads": attn.n_heads,
    "qk_nope_head_dim": attn.qk_nope_head_dim,
    "qk_rope_head_dim": attn.qk_rope_head_dim,
    "v_head_dim": attn.v_head_dim,
    "q_lora_rank": attn.wq_a.out_features,
    "kv_lora_rank": attn.kv_lora_rank,
    "first_k_dense_replace": sum(1 for l in layers if l.moe is None),
    "num_experts": moe.num_experts,
    "n_routed_experts": moe.num_experts,
    "routed_expert_hidden_size": moe.routed_experts.inner_experts.dim,
    "moe_intermediate_size": moe.routed_experts.inner_experts.hidden_dim,
    "attn_res_block_size": layers[0].attn_res_block_size,
}
config_path = MODEL_DIR / "config.json"
cfg = json.loads(config_path.read_text())
for name, value in fields.items():
    before = cfg["text_config"].get(name)
    cfg["text_config"][name] = value
    if before != value:
        print(f"config.json {name}: {before} -> {value}")
lac = cfg["text_config"]["linear_attn_config"]
lac.update({"head_dim": kda.inner_kda.head_dim if hasattr(kda.inner_kda, "head_dim") else 128,
            "num_heads": kda.num_heads if hasattr(kda, "num_heads") else 16,
            "full_attn_layers": full,
            "kda_layers": [i for i in range(1, n + 1) if i not in full]})
print("linear_attn_config:", {k: v for k, v in lac.items()})
config_path.write_text(json.dumps(cfg, indent=1))
check = Path(__file__).with_name("hf_export_shape_check.py")
sys.exit(subprocess.call([sys.executable, str(check), str(MODEL_DIR)]))
