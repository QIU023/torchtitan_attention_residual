"""4025 tree: load our tree's KDA weights by name map, same input, compare."""
import torch
from torchtitan.models.kimi_k3 import _kda_config

P = "/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/xtree_kda.pt"
blob = torch.load(P, map_location="cuda")
ours, x_TD, ref_TD = blob["sd"], blob["x"].cuda(), blob["out"].cuda()

kda = _kda_config(dim=256, num_heads=4, head_dim=32, conv_kernel_size=4)
kda.kernel.lower_bound = None          # our tree's default gate_lower_bound
mod = kda.build().to("cuda").to(torch.float32).eval()

NAME_MAP = {
    "q_proj.weight": "q_proj.weight", "k_proj.weight": "k_proj.weight",
    "v_proj.weight": "v_proj.weight",
    "q_conv1d.weight": "q_conv.weight", "k_conv1d.weight": "k_conv.weight",
    "v_conv1d.weight": "v_conv.weight",
    "f_a_proj.weight": "forget_a.weight", "f_b_proj.weight": "forget_b.weight",
    "b_proj.weight": "beta.weight", "g_proj.weight": "output_gate.weight",
    "o_norm.weight": "output_norm.weight", "o_proj.weight": "output_proj.weight",
    "A_log": "A_log", "dt_bias": "dt_bias",
}
tgt = dict(mod.named_parameters())
missing = []
with torch.no_grad():
    for src, dst in NAME_MAP.items():
        if src not in ours or dst not in tgt:
            missing.append((src, dst)); continue
        v = ours[src].cuda()
        tgt[dst].copy_(v.view_as(tgt[dst]))      # dt_bias is flat on one side
print("unmapped:", missing or "none")
print("theirs-only:", sorted(set(tgt) - set(NAME_MAP.values())) or "none")

with torch.no_grad():
    got = mod(x_TD)
d = (got.float() - ref_TD.float()).abs()
print(f"cross-tree KDA: max_abs={d.max().item():.3e} "
      f"rel={(d.max()/ref_TD.float().abs().max()).item():.3e}")
