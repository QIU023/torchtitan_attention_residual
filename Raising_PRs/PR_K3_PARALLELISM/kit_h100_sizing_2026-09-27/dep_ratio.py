"""Matmul parameters behind DEP's cost ratio: whole vision tower vs one text layer's active parameters (meta).

Active per text layer = everything in the layer except routed experts, plus routed x top_k / num_experts.
"""
import os
from collections import defaultdict

import torch

from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config, model_registry


def counts(config):
    with torch.device("meta"):
        model = config.build()
    tower = sum(p.numel() for n, p in model.named_parameters() if n.startswith("vision_encoder"))
    per_layer = defaultdict(lambda: [0, 0])  # [non-routed, routed]
    for n, p in model.named_parameters():
        if not n.startswith("layers."):
            continue
        i = int(n.split(".")[1])
        routed = ".routed_experts." in n
        per_layer[i][1 if routed else 0] += p.numel()
    return model, tower, per_layer


def report(name, config, top_k, num_experts):
    model, tower, per_layer = counts(config)
    active = [a + r * top_k / num_experts for a, r in per_layer.values()]
    mean = sum(active) / len(active)
    print(f"{name}: tower {tower/1e6:.1f} M, text layers {len(active)}, active per layer mean {mean/1e6:.2f} M "
          f"(min {min(active)/1e6:.2f}, max {max(active)/1e6:.2f}); tower / (3 layers) = {tower/(3*mean):.2f}")


released = model_registry("Kimi-K3", enable_sp=False, seq_len=8192)
report("released Kimi-K3", released, 16, 896)
debug = model_registry("debugmodel", enable_sp=False, seq_len=2048)
report("main debugmodel", debug, 2, 8)
d0917 = _kimi_k3_config(
    enable_sp=False, max_context_length=2048, dim=1024, moe_comm_backend="standard", vocab_size=163840,
    num_layers=24, full_attention_layers={3, 7, 11, 15, 19, 23}, attn_res_block_size=12, num_heads=16,
    q_lora_rank=512, kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64, kda_head_dim=128,
    conv_kernel_size=4, dense_hidden_dim=4096, latent_dim=512, expert_hidden_dim=384, num_experts=32, top_k=4,
    num_shared_experts=2,
    vision_encoder=_vision_encoder_config(text_dim=1024, dim=512, qkv_dim=768, hidden_dim=2048, num_layers=8,
                                          num_heads=6, init_pos_emb_height=32, init_pos_emb_width=32),
    attn_backend="flex")
report("09-17 debugmodel (a3a819c67)", d0917, 4, 32)
for dim in (4096, 5120, 6144):
    probe = _kimi_k3_config(
        enable_sp=False, max_context_length=2048, dim=dim, moe_comm_backend="standard", vocab_size=2048,
        num_layers=93, full_attention_layers={i for i in range(3, 93, 4)}, attn_res_block_size=12, num_heads=16,
        q_lora_rank=512, kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64, kda_head_dim=128,
        conv_kernel_size=4, dense_hidden_dim=2 * dim, latent_dim=256, expert_hidden_dim=256, num_experts=8, top_k=2,
        num_shared_experts=1,
        vision_encoder=_vision_encoder_config(text_dim=dim, dim=1024, qkv_dim=1536, hidden_dim=4096, num_layers=27,
                                              num_heads=12, init_pos_emb_height=64, init_pos_emb_width=64),
        attn_backend="flex")
    report(f"PP probe dim {dim} + released MoonViT", probe, 2, 8)
