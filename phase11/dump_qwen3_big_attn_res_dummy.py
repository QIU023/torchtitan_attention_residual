"""Production-grade Qwen3 + Block AttnRes random-init dummy.

Scaled-up Qwen3 (GQA backbone, no MLA) for 8x 5090 TP=8 prod bench.

Presets (--size):
  qwen3_7b   : L=32 d=4096 H=32 kv_H=8 intermediate=11008 → ~7B (30% fill of 8x 5090)
  qwen3_14b  : L=40 d=5120 H=40 kv_H=8 intermediate=13824 → ~14B (55% fill)
  qwen3_32b  : L=64 d=5120 H=40 kv_H=8 intermediate=27648 → ~32B (TP=8 needed)

Why Qwen3 + AttnRes for prod multi-card bench:
  - GQA backbone (no MLA) → no flashinfer_mla NaN on Blackwell
  - Generic AttnRes overlay (sibling of Kimi overlay, proves overlay
    architecture is reusable)
  - Standard ROPE attention, easier baseline to interpret

Run::

    python3 phase11/dump_qwen3_big_attn_res_dummy.py \
        --size qwen3_14b \
        --out phase11/hf_qwen3_14b_attn_res
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


_PRESETS = {
    # Qwen3 official sizes (from Qwen3Config defaults + scaling)
    "qwen3_7b": dict(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, intermediate_size=11008,
        attn_res_num_blocks=8,  # 4 transformer-blocks per AttnRes-block
    ),
    "qwen3_14b": dict(
        hidden_size=5120, num_hidden_layers=40, num_attention_heads=40,
        num_key_value_heads=8, head_dim=128, intermediate_size=13824,
        attn_res_num_blocks=10,  # 4 transformer-blocks per AttnRes-block
    ),
    "qwen3_32b": dict(
        hidden_size=5120, num_hidden_layers=64, num_attention_heads=40,
        num_key_value_heads=8, head_dim=128, intermediate_size=27648,
        attn_res_num_blocks=16,  # 4 transformer-blocks per AttnRes-block
    ),
}


def _bootstrap_single_proc():
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=os.environ.get("MASTER_PORT", "29573"),
        RANK="0", WORLD_SIZE="1", LOCAL_RANK="0",
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    from sglang.srt.distributed import (
        init_distributed_environment, initialize_model_parallel,
    )
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    import sglang.srt.layers.dp_attention as _dp_attn
    _dp_attn._ATTN_DP_SIZE = 1
    _dp_attn._LOCAL_ATTN_DP_SIZE = 1
    _dp_attn._ATTN_DP_RANK = 0
    _dp_attn._LOCAL_ATTN_DP_RANK = 0
    _dp_attn._ENABLE_DP_ATTENTION_FLAG = False


def _make_qwen3_big_attn_res_config(preset: dict):
    from transformers import Qwen3Config

    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=preset["hidden_size"],
        intermediate_size=preset["intermediate_size"],
        num_hidden_layers=preset["num_hidden_layers"],
        num_attention_heads=preset["num_attention_heads"],
        num_key_value_heads=preset["num_key_value_heads"],
        head_dim=preset["head_dim"],
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    cfg.attn_res_num_blocks = preset["attn_res_num_blocks"]
    cfg.architectures = ["Qwen3BlockAttnResForCausalLM"]
    cfg.torch_dtype = "bfloat16"
    return cfg


def _build_model(cfg):
    from sglang.srt.models.qwen3_attn_res_overlay import Qwen3BlockAttnResForCausalLM
    with torch.device("cpu"):
        m = Qwen3BlockAttnResForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.numel() == 0 or not p.dtype.is_floating_point:
                continue
            if "attn_res_proj" in n or "final_attn_res_proj" in n:
                continue  # zero-init per paper §5
            if p.abs().max().item() == 0:
                p.normal_(mean=0.0, std=0.02)
    return m


def _live_to_hf_state_dict(model, cfg) -> dict:
    """Same logic as the 120M smoke dumper — split fused qkv + gate_up."""
    raw = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    seen_storages: set[int] = set()

    def emit(name: str, tensor: torch.Tensor):
        if tensor.numel() == 0:
            return
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in seen_storages:
            return
        seen_storages.add(ptr)
        out[name] = tensor.detach().cpu().contiguous()

    H = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    D = cfg.head_dim if cfg.head_dim else cfg.hidden_size // H
    intermediate = cfg.intermediate_size

    for k, v in raw.items():
        if v.ndim == 0:
            continue
        if k.endswith(".self_attn.qkv_proj.weight"):
            base = k.replace(".qkv_proj.weight", "")
            q_size = H * D
            kv_size = H_kv * D
            emit(f"{base}.q_proj.weight", v.narrow(0, 0, q_size).clone())
            emit(f"{base}.k_proj.weight", v.narrow(0, q_size, kv_size).clone())
            emit(f"{base}.v_proj.weight", v.narrow(0, q_size + kv_size, kv_size).clone())
            continue
        if k.endswith(".mlp.gate_up_proj.weight"):
            base = k.replace(".gate_up_proj.weight", "")
            emit(f"{base}.gate_proj.weight", v.narrow(0, 0, intermediate).clone())
            emit(f"{base}.up_proj.weight", v.narrow(0, intermediate, intermediate).clone())
            continue
        emit(k, v)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", choices=list(_PRESETS.keys()), default="qwen3_14b")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _bootstrap_single_proc()
    preset = _PRESETS[args.size]
    cfg = _make_qwen3_big_attn_res_config(preset)
    print(
        f"[1/3] {args.size}: L={cfg.num_hidden_layers} d={cfg.hidden_size} "
        f"H={cfg.num_attention_heads}/{cfg.num_key_value_heads} (GQA) "
        f"ff={cfg.intermediate_size} AttnRes N={cfg.attn_res_num_blocks}",
        flush=True,
    )

    m = _build_model(cfg)
    total = sum(p.numel() for p in m.parameters())
    bytes_est = total * 2
    print(f"[2/3] total params: {total:,} (~{total / 1e9:.1f} B); "
          f"safetensors ~{bytes_est / 1e9:.1f} GB", flush=True)

    sd = _live_to_hf_state_dict(m, cfg)
    from safetensors.torch import save_file
    save_file(sd, str(out / "model.safetensors"))
    print(f"[3/3] safetensors: {len(sd)} tensors -> {out / 'model.safetensors'}", flush=True)

    cfg_d = cfg.to_dict()
    with open(out / "config.json", "w") as f:
        json.dump(cfg_d, f, indent=2, default=str)
    print(f"      config.json -> {out / 'config.json'}", flush=True)


if __name__ == "__main__":
    main()
