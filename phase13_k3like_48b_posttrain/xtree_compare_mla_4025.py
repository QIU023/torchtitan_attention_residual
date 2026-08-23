"""4025 tree: our tree's MLA weights, same input."""
import torch
from torchtitan.models.kimi_k3 import _mla_config
from torchtitan.models.common.attention import create_attention_mask, get_causal_mask_mod

P = "/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/xtree_mla.pt"
blob = torch.load(P, map_location="cuda")
ours, x_TD, ref_TD = blob["sd"], blob["x"].cuda(), blob["out"].cuda()

cfg = _mla_config(dim=256, num_heads=4, q_lora_rank=64, kv_lora_rank=64,
                  qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
                  attn_backend="flex")
mod = cfg.build().to("cuda").to(torch.float32).eval()

NAME_MAP = {
    "q_a_proj.weight": "wq_a.weight", "q_a_layernorm.weight": "q_norm.weight",
    "q_b_proj.weight": "wq_b.weight",
    "kv_a_proj_with_mqa.weight": "wkv_a.weight",
    "kv_a_layernorm.weight": "kv_norm.weight", "kv_b_proj.weight": "wkv_b.weight",
    "attn_gate_proj.weight": "gate.weight", "o_proj.weight": "wo.weight",
}
tgt = dict(mod.named_parameters())
miss = [(s, d) for s, d in NAME_MAP.items() if s not in ours or d not in tgt]
with torch.no_grad():
    for s, d in NAME_MAP.items():
        if (s, d) not in miss:
            tgt[d].copy_(ours[s].cuda().view_as(tgt[d]))
print("unmapped:", miss or "none")
print("theirs-only:", sorted(set(tgt) - set(NAME_MAP.values())) or "none")

T = x_TD.shape[0]
mask = create_attention_mask(get_causal_mask_mod(), None, None, T, T, device="cuda")
with torch.no_grad():
    got = mod(x_TD, attention_masks=mask)
d = (got.float() - ref_TD.float()).abs()
print(f"cross-tree MLA: max_abs={d.max().item():.3e} "
      f"rel={(d.max()/ref_TD.float().abs().max()).item():.3e}")
