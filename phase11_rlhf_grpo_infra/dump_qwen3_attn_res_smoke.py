"""Tiny Qwen3 + Block AttnRes random-init dummy ckpt for SGLang smoke.

Generality test: this file is the proof that
``sglang/srt/layers/attn_res.py`` works as a reusable algorithm and
``models/<model>_attn_res_overlay.py`` works as a reusable wrapper
pattern. We pick a small Qwen3 dense shape (no MoE, no MLA) as the
maximally-different second carrier vs the Kimi Linear overlay.

Shape:
* L=8, d=512, num_heads=8, num_kv_heads=4 (GQA 2:1), head_dim=64
* intermediate_size=1024, vocab=151936 (Qwen3 default)
* attn_res_num_blocks=4 → layers_per_block=2

~120M param dummy — fast to dump + boot under any TP/PP/EP config.

Run::

    python3 phase11_rlhf_grpo_infra/dump_qwen3_attn_res_smoke.py --out phase11_rlhf_grpo_infra/hf_qwen3_attn_res
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _bootstrap_single_proc():
    """Init dist + sglang model-parallel + global server args (CPU dump)."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=os.environ.get("MASTER_PORT", "29570"),
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

    # Qwen3DecoderLayer constructs a ``LayerCommunicator`` which calls
    # ``get_attention_dp_size()``; that asserts the DP-attn module
    # globals are set. The full ``initialize_dp_attention`` requires a
    # real ``ModelConfig`` which we don't have for a standalone dump.
    # For a single-process CPU dump (dp_size=1, no DP attention), the
    # globals are trivial — set them directly.
    import sglang.srt.layers.dp_attention as _dp_attn
    _dp_attn._ATTN_DP_SIZE = 1
    _dp_attn._LOCAL_ATTN_DP_SIZE = 1
    _dp_attn._ATTN_DP_RANK = 0
    _dp_attn._LOCAL_ATTN_DP_RANK = 0
    _dp_attn._ENABLE_DP_ATTENTION_FLAG = False


def _make_qwen3_attn_res_config():
    """Build a Qwen3Config-like config with attn_res fields tacked on.

    We use the transformers Qwen3Config (which sglang's qwen3.py imports
    via ``Qwen3Config = None`` and then resolved at runtime through HF
    AutoConfig). For our smoke we instantiate it directly.
    """
    from transformers import Qwen3Config

    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=True,
    )
    # AttnRes-specific extra; consumed via getattr in the overlay model.
    cfg.attn_res_num_blocks = 4
    cfg.architectures = ["Qwen3BlockAttnResForCausalLM"]
    cfg.torch_dtype = "bfloat16"
    return cfg


def _build_model(cfg):
    """Construct overlay model on CPU; do NOT cast dtype globally.

    The SGLang loader picks each param's natural dtype from the live
    construction context; saving in init dtypes (mostly float32) lets the
    loader cast on ``copy_`` per its own rules.
    """
    from sglang.srt.models.qwen3_attn_res_overlay import (
        Qwen3BlockAttnResForCausalLM,
    )
    with torch.device("cpu"):
        m = Qwen3BlockAttnResForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.numel() == 0 or not p.dtype.is_floating_point:
                continue
            # Zero-init pseudo-queries (paper § 5).
            if "attn_res_proj" in n or "final_attn_res_proj" in n:
                continue
            if p.abs().max().item() == 0:
                p.normal_(mean=0.0, std=0.02)
    return m


def _live_to_hf_state_dict(model, cfg) -> dict:
    """Walk live state_dict, emit HF unfused naming.

    Qwen3 live params include fused ``self_attn.qkv_proj.weight``
    and ``mlp.gate_up_proj.weight``. The upstream loader fuses unfused
    HF tensors via ``stacked_params_mapping``, so we split the live
    fused weights back into the HF names.
    """
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

        # Split self_attn.qkv_proj.weight → q_proj/k_proj/v_proj
        # qkv_proj weight shape: [(H + 2*H_kv) * D, hidden]
        if k.endswith(".self_attn.qkv_proj.weight"):
            base = k.replace(".qkv_proj.weight", "")
            q_size = H * D
            kv_size = H_kv * D
            emit(f"{base}.q_proj.weight",
                 v.narrow(0, 0, q_size).clone())
            emit(f"{base}.k_proj.weight",
                 v.narrow(0, q_size, kv_size).clone())
            emit(f"{base}.v_proj.weight",
                 v.narrow(0, q_size + kv_size, kv_size).clone())
            continue

        # Split mlp.gate_up_proj.weight → gate_proj + up_proj
        # gate_up_proj weight shape: [2 * intermediate, hidden]
        if k.endswith(".mlp.gate_up_proj.weight"):
            base = k.replace(".gate_up_proj.weight", "")
            emit(f"{base}.gate_proj.weight",
                 v.narrow(0, 0, intermediate).clone())
            emit(f"{base}.up_proj.weight",
                 v.narrow(0, intermediate, intermediate).clone())
            continue

        # Pass-through (layernorms, q_norm/k_norm, o_proj, down_proj,
        # attn_res_*, mlp_res_*, embed_tokens, norm, lm_head, etc.)
        emit(k, v)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _bootstrap_single_proc()
    cfg = _make_qwen3_attn_res_config()
    print(
        f"[1/3] Qwen3 AttnRes cfg: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"H={cfg.num_attention_heads} kv_H={cfg.num_key_value_heads} head_dim={cfg.head_dim}"
    )
    m = _build_model(cfg)
    print(f"[2/3] model: {sum(p.numel() for p in m.parameters()):,} params")

    sd = _live_to_hf_state_dict(m, cfg)
    from safetensors.torch import save_file
    save_file(sd, str(out / "model.safetensors"))
    print(f"[3/3] safetensors: {len(sd)} tensors -> {out/'model.safetensors'}")

    cfg_d = cfg.to_dict()
    with open(out / "config.json", "w") as f:
        json.dump(cfg_d, f, indent=2, default=str)
    print(f"      config.json -> {out/'config.json'}")


if __name__ == "__main__":
    main()
