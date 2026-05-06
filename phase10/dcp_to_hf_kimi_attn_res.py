"""DCP → HF safetensors conversion for kimi_linear (Block AttnRes).

Loads our torchtitan kimi_linear DCP-sharded checkpoint and writes
HF-format safetensors that SGLang's ``kimi_linear`` model + a thin
AttnRes extension can load.

Mapping summary
---------------
1. Standard transformer blocks (input/post LNs, self_attn, mlp): map
   to HF naming with `model.` prefix and `ffn` -> `mlp`.
2. MoE experts: fused `[E, I, H]` tensor split into per-expert
   `gate_proj/up_proj/down_proj`. w1 -> gate_proj, w3 -> up_proj,
   w2 -> down_proj (HF/HF-mirror Kimi convention).
3. AttnRes-specific params (4 per layer + 2 final): kept under same
   relative names with `model.` prefix. SGLang's stock kimi_linear
   does not have these; the extension model registers them.

Run (single-rank DCP load is enough — HF format is replicated):
    torchrun --nproc_per_node=1 phase10/dcp_to_hf_kimi_attn_res.py \\
        --in phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000 \\
        --out phase10/hf_ckpt_phase4_step8000 \\
        --config kimi_linear_436m_block_attn_res_n4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))
sys.path.insert(0, str(_WS / "torchtitan"))


# ---- key remapping ---------------------------------------------------------
# w1 = gate_proj, w3 = up_proj, w2 = down_proj  (Kimi/HF convention)
_W_TO_HF = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def remap_one(name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Map one DCP key to (possibly multiple) HF keys + tensors.

    Returns a list of (hf_name, tensor) pairs. Multiple emissions
    occur for fused-MoE experts which split into per-expert linears.
    """
    # Top-level
    if name == "embed_tokens.weight":
        return [("model.embed_tokens.weight", tensor)]
    if name == "norm.weight":
        return [("model.norm.weight", tensor)]
    if name == "lm_head.weight":
        return [("lm_head.weight", tensor)]
    if name == "final_attn_res_proj.weight":
        return [("model.final_attn_res_proj.weight", tensor)]
    if name == "final_attn_res_norm.weight":
        return [("model.final_attn_res_norm.weight", tensor)]

    # layers.{i}....
    if name.startswith("layers."):
        rest = name[len("layers.") :]
        idx_s, _, sub = rest.partition(".")
        prefix = f"model.layers.{idx_s}"

        # AttnRes-specific (preserve)
        for tag in ("attn_res_proj", "attn_res_norm", "mlp_res_proj", "mlp_res_norm"):
            if sub == f"{tag}.weight":
                return [(f"{prefix}.{tag}.weight", tensor)]

        # Layernorms
        if sub in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            return [(f"{prefix}.{sub}", tensor)]

        # self_attn.* — names already match (q_proj, kv_a_proj_with_mqa, kv_b_proj,
        # o_proj, kv_a_layernorm, A_log, b_proj, dt_bias, f_a_proj, f_b_proj,
        # g_a_proj, g_b_proj, k_conv1d, k_proj, o_norm, q_conv1d, v_conv1d, v_proj,
        # etc.).
        if sub.startswith("self_attn."):
            return [(f"{prefix}.{sub}", tensor)]

        # Dense MLP (layer 0): ffn.gate_proj/up_proj/down_proj.weight
        if sub.startswith("ffn.gate_proj") or sub.startswith("ffn.up_proj") \
                or sub.startswith("ffn.down_proj"):
            return [(f"{prefix}.mlp.{sub[4:]}", tensor)]

        # MoE structure: ffn._moe.{...}
        if sub.startswith("ffn._moe."):
            inner = sub[len("ffn._moe.") :]

            # Router gate: ffn._moe.router.gate.weight -> mlp.gate.weight
            if inner == "router.gate.weight":
                return [(f"{prefix}.mlp.gate.weight", tensor)]

            # Expert correction bias
            if inner == "expert_bias":
                return [(f"{prefix}.mlp.gate.e_score_correction_bias", tensor)]

            # Fused experts: experts.{w1,w2,w3} of shape [E, I, H] (or [E, H, I] for w2)
            if inner.startswith("experts."):
                w_tag = inner[len("experts.") :]  # "w1" | "w2" | "w3"
                hf_tag = _W_TO_HF.get(w_tag)
                if hf_tag is None:
                    raise ValueError(f"Unknown expert weight tag: {w_tag!r} in {name!r}")
                # tensor shape [E, A, B] — split into E per-expert tensors of shape [A, B]
                E = tensor.shape[0]
                return [
                    (f"{prefix}.mlp.experts.{e}.{hf_tag}.weight", tensor[e].contiguous())
                    for e in range(E)
                ]

            # Shared experts: shared_experts.{w1,w2,w3}.weight -> mlp.shared_experts.{*}.weight
            if inner.startswith("shared_experts."):
                tail = inner[len("shared_experts.") :]
                w_tag, _, suff = tail.partition(".")  # "w1" + "weight"
                hf_tag = _W_TO_HF.get(w_tag)
                if hf_tag is None:
                    raise ValueError(f"Unknown shared expert tag: {w_tag!r}")
                return [(f"{prefix}.mlp.shared_experts.{hf_tag}.{suff}", tensor)]

        raise ValueError(f"Unmapped layer key: layers.{idx_s}.{sub}")

    raise ValueError(f"Unmapped top-level key: {name!r}")


def build_skeleton_state_dict(config_name: str):
    from torchtitan.experiments.kimi_linear.config_registry import (
        kimi_linear_436m_block_attn_res_n4,
    )
    from torchtitan.experiments.kimi_linear.attn_res_model import (
        KimiLinearAttnResModel,
    )
    if config_name != "kimi_linear_436m_block_attn_res_n4":
        raise NotImplementedError(f"Only 436m_block_attn_res_n4 supported now, got {config_name}")
    cfg = kimi_linear_436m_block_attn_res_n4()
    spec = cfg.model_spec.model
    model = KimiLinearAttnResModel(spec.kimi_config, num_blocks=spec.num_blocks)
    model.init_weights()
    return spec.kimi_config, model.state_dict()


def make_hf_config(kimi_config) -> dict:
    """Produce HF config.json content matching SGLang's KimiLinearConfig.

    Kimi K-series models load via trust_remote_code in HF, so the
    architectures field references our wrapper class. SGLang's
    KimiLinearConfig is a PretrainedConfig accepting these kwargs.
    """
    return {
        "architectures": ["KimiBlockAttnResForCausalLM"],
        "model_type": "kimi_linear",
        # core dims
        "hidden_size": kimi_config.hidden_size,
        "intermediate_size": kimi_config.intermediate_size,
        "num_hidden_layers": kimi_config.num_hidden_layers,
        "num_attention_heads": kimi_config.num_attention_heads,
        "num_key_value_heads": kimi_config.num_key_value_heads,
        "head_dim": kimi_config.head_dim,
        "vocab_size": kimi_config.vocab_size,
        "hidden_act": kimi_config.hidden_act,
        "rms_norm_eps": kimi_config.rms_norm_eps,
        "tie_word_embeddings": getattr(kimi_config, "tie_word_embeddings", True),
        # MLA
        "kv_lora_rank": kimi_config.kv_lora_rank,
        "qk_nope_head_dim": kimi_config.qk_nope_head_dim,
        "qk_rope_head_dim": kimi_config.qk_rope_head_dim,
        "v_head_dim": kimi_config.v_head_dim,
        "q_lora_rank": getattr(kimi_config, "q_lora_rank", None),
        "mla_use_nope": kimi_config.mla_use_nope,
        # MoE
        "is_moe": kimi_config.is_moe,
        "is_mla": kimi_config.is_mla,
        "num_experts": kimi_config.num_experts,
        "n_routed_experts": kimi_config.num_experts,
        "num_experts_per_token": kimi_config.num_experts_per_token,
        "num_shared_experts": kimi_config.num_shared_experts,
        "moe_intermediate_size": kimi_config.moe_intermediate_size,
        "moe_renormalize": kimi_config.moe_renormalize,
        "moe_router_activation_func": kimi_config.moe_router_activation_func,
        "moe_layer_freq": kimi_config.moe_layer_freq,
        "first_k_dense_replace": kimi_config.first_k_dense_replace,
        "num_expert_group": kimi_config.num_expert_group,
        "topk_group": getattr(kimi_config, "topk_group", 1),
        "routed_scaling_factor": getattr(kimi_config, "routed_scaling_factor", 1.0),
        "use_grouped_topk": getattr(kimi_config, "use_grouped_topk", False),
        # KDA
        "kda_layers": list(kimi_config.kda_layers),
        "full_attn_layers": list(kimi_config.full_attn_layers),
        "kda_num_heads": kimi_config.kda_num_heads,
        "kda_head_dim": kimi_config.kda_head_dim,
        "kda_short_conv_kernel_size": kimi_config.kda_short_conv_kernel_size,
        # AttnRes (custom — used by our extension)
        "attn_res_enabled": True,
        "attn_res_num_blocks": 4,
    }


def init_dist():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, type=Path)
    p.add_argument("--out", dest="out_dir", required=True, type=Path)
    p.add_argument("--config", default="kimi_linear_436m_block_attn_res_n4")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    init_dist()
    torch.cuda.set_device(0)

    print(f"[conv] building skeleton state dict for config={args.config}")
    kimi_config, sd = build_skeleton_state_dict(args.config)
    print(f"[conv] skeleton has {len(sd)} keys")

    print(f"[conv] loading DCP from {args.in_dir}")
    dcp.load(sd, checkpoint_id=str(args.in_dir))

    target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]

    print(f"[conv] remapping {len(sd)} keys + casting to {target_dtype}")
    hf_sd: dict[str, torch.Tensor] = {}
    n_in = 0
    for name, t in sd.items():
        n_in += 1
        for hf_name, hf_t in remap_one(name, t):
            hf_sd[hf_name] = hf_t.detach().to(target_dtype).contiguous()

    print(f"[conv] DCP keys in: {n_in}, HF keys out: {len(hf_sd)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Write config.json
    cfg_dict = make_hf_config(kimi_config)
    with (args.out_dir / "config.json").open("w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[conv] wrote config.json")

    # Write safetensors (single shard for now — model is 436M ≈ 870 MB at bf16)
    from safetensors.torch import save_file

    save_file(hf_sd, str(args.out_dir / "model.safetensors"))
    total_bytes = sum(t.element_size() * t.numel() for t in hf_sd.values())
    print(f"[conv] wrote model.safetensors ({total_bytes / 1024**2:.1f} MB)")

    # Write a manifest for sanity / debugging
    manifest = {
        "source_dcp": str(args.in_dir),
        "n_dcp_keys": n_in,
        "n_hf_keys": len(hf_sd),
        "dtype": args.dtype,
        "total_bytes": total_bytes,
        "key_sample": sorted(hf_sd.keys())[:20] + ["..."] + sorted(hf_sd.keys())[-5:],
    }
    with (args.out_dir / "conversion_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[conv] DONE → {args.out_dir}")


if __name__ == "__main__":
    main()
