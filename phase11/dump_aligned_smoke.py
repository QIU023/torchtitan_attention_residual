"""Canonical aligned design point for the SGLang-served AttnRes model.

This is the model size we run AttnRes inference on. It is chosen by:

* **Hardware fit**: 8× RTX 5090 32 GB. We want SGLang 3D parallel
  inference (TP=2 × PP=2 × EP=2 = 8 ranks), so each rank holds
  ``total_params / 8`` weights plus ~27 GB of KV-cache pool. Targeting
  ~1.4 B total bf16 (~2.8 GB) leaves ample headroom on every rank.
* **Aligned dims**: every head/per-channel dim must be a multiple of 8
  (and qk_rope must be ≥ 16) for flashinfer's batch-prefill kernel to
  accept the layout on SM 12.0. ``d/num_heads = 64`` and
  ``qk_nope=64, qk_rope=32, v=64`` give ``head_dim_qk=96, head_dim_vo=64``,
  both 8/16/32-aligned.
* **Activated-param parity with phase4 436M**: 32 routed experts top-8
  with ``moe_intermediate_size=768`` lands at ~447 M activated /
  ~1.4 B total — same order of magnitude as the phase4 scaling-law
  row (436M activated, ~1.4 B total) but with SGLang-friendly head
  dims (vs phase4's 73/36/73 which break flashinfer at SM 12.0).
* **Re-trainable locally**: ~1.4 B params on 8× 5090 takes ~hours to
  reach a coherent loss curve, so the user can optionally re-train
  this exact shape end-to-end.

The deliverable using this shape is **AttnRes parallel-inference** in
SGLang — the model serves only as a carrier to demonstrate the
generalised AttnRes overlay running through SGLang's TP/PP/EP fabric.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from dump_dummy_hf_ckpt import (  # noqa: E402
    _bootstrap_single_proc,
    _build_model,
    _live_to_hf_state_dict,
)


def _make_aligned_config():
    from sglang.srt.configs.kimi_linear import KimiLinearConfig

    n_layers = 16
    d = 1024
    H = 16

    qk_nope = 64
    qk_rope = 32
    v_head = 64
    kda_head_dim = 64
    kv_lora_rank = 512   # multiple of 64
    # ff=768 hits ~447M activated (top-8/32 routed + 1 shared expert),
    # close to phase4's 436M activated row. Multiple of 64 → kernel-friendly.
    d_ff = 768

    # KDA:MLA = 3:1 → MLA on every 4th layer (1-indexed)
    period = 4
    kda_layers = [i for i in range(1, n_layers + 1) if i % period != 0]
    full_attn_layers = [i for i in range(1, n_layers + 1) if i % period == 0]

    return KimiLinearConfig(
        vocab_size=163840,
        hidden_size=d,
        tie_word_embeddings=True,
        num_hidden_layers=n_layers,
        intermediate_size=d_ff,
        num_attention_heads=H,
        num_key_value_heads=H,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_head,
        mla_use_nope=True,
        linear_attn_config={
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            "num_heads": H,
            "head_dim": kda_head_dim,
            "short_conv_kernel_size": 4,
        },
        num_experts=32,
        num_experts_per_token=8,
        moe_intermediate_size=d_ff,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=1,
        routed_scaling_factor=2.446,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        attn_res_num_blocks=4,
        architectures=[
            "KimiBlockAttnResForCausalLM",
            "KimiLinearForCausalLM",
        ],
        torch_dtype="bfloat16",
        hidden_act="silu",
        initializer_range=0.02,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _bootstrap_single_proc()
    cfg = _make_aligned_config()
    print(
        f"[1/3] aligned config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"qk_nope={cfg.qk_nope_head_dim} qk_rope={cfg.qk_rope_head_dim} v={cfg.v_head_dim}"
    )
    m = _build_model(cfg)
    print(f"[2/3] model: {sum(p.numel() for p in m.parameters()):,} params")

    sd = _live_to_hf_state_dict(m, cfg)
    from safetensors.torch import save_file
    save_file(sd, str(out / "model.safetensors"))
    print(f"[3/3] safetensors: {len(sd)} tensors -> {out/'model.safetensors'}")

    cfg_d = cfg.to_dict()
    with open(out / "config.json", "w") as f:
        json.dump(cfg_d, f, indent=2)
    print(f"      config.json -> {out/'config.json'}")


if __name__ == "__main__":
    main()
