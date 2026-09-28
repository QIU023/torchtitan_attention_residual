"""The current debug model widened by s: every width that is a multiple of dim in the debug model
scales with it (heads by count, head dims fixed); layers, blocks, experts, top-k and vocab unchanged.

Usage: python widen_debug.py <pp> <vp> <dim> [<dim> ...]   (meta only)
Static = 16 bytes per parameter (fp32 weight, fp32 grad, two AdamW states); stage s on rank s % pp.
"""
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch

from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part
from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config

GiB = 2**30


def widened(dim, seq=2048):
    s = dim // 256
    assert dim == 256 * s
    return _kimi_k3_config(
        enable_sp=False, max_context_length=seq, dim=dim, moe_comm_backend="standard", vocab_size=2048,
        num_layers=17, full_attention_layers={3, 7, 11, 15, 16}, attn_res_block_size=4,
        num_heads=4 * s, q_lora_rank=128 * s, kv_lora_rank=64 * s,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64, kda_head_dim=128, conv_kernel_size=4,
        dense_hidden_dim=512 * s, latent_dim=128 * s, expert_hidden_dim=128 * s,
        num_experts=8, top_k=2, num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim, dim=256 * s, qkv_dim=512 * s, hidden_dim=512 * s, num_layers=2, num_heads=4 * s,
            init_pos_emb_height=32, init_pos_emb_width=32),
        attn_backend="flex")


pp, vp = int(sys.argv[1]), int(sys.argv[2])
for dim in [int(x) for x in sys.argv[3:]]:
    with torch.device("meta"):
        model = widened(dim).build()
    per_module = defaultdict(int)
    for name, p in model.named_parameters():
        parts = name.split(".")
        per_module[".".join(parts[:2]) if parts[0] == "layers" else parts[0]] += p.numel()
    total = sum(per_module.values())
    split = _generate_llm_fqn_per_model_part(pp * vp, 17, 1, 1)
    placed = {m for part in split for m in part}
    per_rank = defaultdict(int)
    for st, part in enumerate(split):
        for m in part:
            per_rank[st % pp] += per_module.get(m, 0)
    for k in per_module:
        if k not in placed:
            per_rank[0 if "vision" in k else (pp * vp - 1) % pp] += per_module[k]
    tower = per_module.get("vision_encoder", 0)
    print(f"dim {dim} (x{dim // 256}): {total / 1e9:.3f} B params (tower {tower / 1e6:.1f} M), "
          f"one GPU static {16 * total / GiB:.1f} GiB; pp{pp} x vp{vp} static per rank "
          + " ".join(f"{16 * per_rank[r] / GiB:.1f}" for r in range(pp)))
