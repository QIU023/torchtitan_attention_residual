"""Production-grade Kimi Linear 48B-layout + AttnRes random-init dummy.

Paper layout (`kimi_linear_48b_block_attn_res` from torchtitan registry,
matching MoonshotAI/Kimi-Linear-48B-A3B-Base config.json):

* 27 transformer-blocks (= 54 paper-layers)
* d_model = 2304, num_heads = 32
* Dense FFN intermediate = 9216 (layer 0 only)
* MoE: 256 experts (downscalable via --num-experts), top-8, 1 shared, moe_intermediate = 1024
* MLA: kv_lora_rank=512, qk_nope=128, qk_rope=64, v_head=128 (paper-exact)
* KDA: head_dim=128, 20 layers; MLA on 7 layers — paper pattern
* AttnRes: num_blocks=9 (3 transformer-blocks per AttnRes-block, paper sweet-spot)

Param counts (bf16 weight bytes for safetensors):
  --num-experts 256 (paper) : ~49B params, ~98 GB safetensors
  --num-experts 64           : ~14B params, ~28 GB safetensors
  --num-experts 32           : ~7B params, ~14 GB safetensors
  --num-experts 16           : ~4B params, ~8 GB safetensors

For 8× RTX 5090 prod density bench:
  num-experts=32 → ~7B model, fills ~25% of 256 GB (8× 32GB)
  num-experts=64 → ~14B model, fills ~50%
  num-experts=128 → ~28B model, fills 80% (near-paper density)

Run::

    python3 phase11/dump_kimi_48b_attn_res_dummy.py \
        --num-experts 64 \
        --out phase11/hf_kimi_48b_attn_res_e64
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _bootstrap_single_proc():
    """Init torch.distributed + sglang model-parallel for single-proc dump.

    Uses gloo backend so no GPU is touched during weight dump (matches
    `dump_dummy_hf_ckpt.py`).
    """
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=os.environ.get("MASTER_PORT", "29572"),
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
    )
    from sglang.srt.server_args import (
        ServerArgs, set_global_server_args_for_scheduler,
    )
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    import sglang.srt.layers.dp_attention as _dp_attn
    _dp_attn._ATTN_DP_SIZE = 1
    _dp_attn._LOCAL_ATTN_DP_SIZE = 1
    _dp_attn._ATTN_DP_RANK = 0
    _dp_attn._LOCAL_ATTN_DP_RANK = 0
    _dp_attn._ENABLE_DP_ATTENTION_FLAG = False


def _make_kimi_48b_attn_res_config(num_experts: int) -> "KimiLinearConfig":
    """Paper-exact Kimi 48B-A3B layout, MoE expert count parameterized.

    Mirrors `kimi_linear_48b_block_attn_res` flavor (torchtitan
    `experiments/kimi_linear/config_registry.py`), translated to
    SGLang's nested `linear_attn_config` schema.
    """
    from sglang.srt.configs.kimi_linear import KimiLinearConfig

    # Paper §"Training recipe" exact:
    n_layers = 27
    d = 2304
    H = 32

    # Paper-exact MLA dims (not derived from d/H ratios)
    head_dim_mla_nope = 128
    head_dim_mla_rope = 64
    head_dim_mla_v = 128
    kda_head_dim = 128
    kv_lora_rank = 512

    # Paper KDA/MLA layer pattern (1-indexed, from HF config.json)
    kda_layers = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                  17, 18, 19, 21, 22, 23, 25, 26]
    full_attn_layers = [4, 8, 12, 16, 20, 24, 27]

    cfg = KimiLinearConfig(
        vocab_size=163840,
        hidden_size=d,
        tie_word_embeddings=False,  # paper: lm_head separate from embedding
        num_hidden_layers=n_layers,
        # Dense FFN at layer 0 (first_k_dense_replace=1); paper uses 9216
        intermediate_size=9216,
        num_attention_heads=H,
        num_key_value_heads=H,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        # MLA — paper-exact
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=head_dim_mla_nope,
        qk_rope_head_dim=head_dim_mla_rope,
        v_head_dim=head_dim_mla_v,
        mla_use_nope=True,
        # KDA — paper layer pattern
        linear_attn_config={
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            "num_heads": H,
            "head_dim": kda_head_dim,
            "short_conv_kernel_size": 4,
        },
        # MoE — paper has 256 experts; downscale for compute budget
        num_experts=num_experts,
        num_experts_per_token=8,
        moe_intermediate_size=1024,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=1,
        routed_scaling_factor=2.446,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        # Paper-exact Block AttnRes: N=9 (3 transformer-blocks per
        # AttnRes-block = 6 paper-layers per AttnRes-block)
        attn_res_num_blocks=9,
        architectures=[
            "KimiBlockAttnResForCausalLM",
            "KimiLinearForCausalLM",
        ],
        torch_dtype="bfloat16",
        hidden_act="silu",
        initializer_range=0.02,
    )
    return cfg


def _build_model(cfg):
    """Reuse dump_dummy_hf_ckpt's model builder."""
    from sglang.srt.models.attn_res_overlay import (
        KimiBlockAttnResForCausalLM,
    )
    with torch.device("cpu"):
        m = KimiBlockAttnResForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.numel() == 0 or not p.dtype.is_floating_point:
                continue
            # Paper §5: zero-init pseudo-queries
            if "attn_res_proj" in n or "final_attn_res_proj" in n:
                continue
            if p.abs().max().item() == 0:
                p.normal_(mean=0.0, std=0.02)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output HF ckpt dir")
    ap.add_argument(
        "--num-experts", type=int, default=32,
        help="paper=256; default 32 = ~7B params (8x 5090 light); 64 = ~14B (50%% fill); 128 = ~28B (80%% fill)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _bootstrap_single_proc()
    cfg = _make_kimi_48b_attn_res_config(num_experts=args.num_experts)
    print(
        f"[1/3] Kimi 48B-layout cfg: L={cfg.num_hidden_layers} "
        f"d={cfg.hidden_size} H={cfg.num_attention_heads} "
        f"experts={cfg.num_experts}({cfg.num_experts_per_token}) "
        f"AttnRes N={cfg.attn_res_num_blocks}",
        flush=True,
    )

    # Re-use the live→HF state_dict translator from the 436M dumper
    # (same KimiBlockAttnResForCausalLM class, same fused-tensor schema).
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from dump_dummy_hf_ckpt import _live_to_hf_state_dict

    m = _build_model(cfg)
    total = sum(p.numel() for p in m.parameters())
    print(f"[2/3] total params: {total:,} (~{total / 1e9:.1f} B)", flush=True)

    sd = _live_to_hf_state_dict(m, cfg)
    # Estimate safetensors size (bf16 = 2 bytes per param)
    bytes_est = sum(t.numel() * 2 for t in sd.values())
    print(f"      safetensors size estimate: ~{bytes_est / 1e9:.1f} GB", flush=True)

    from safetensors.torch import save_file
    save_file(sd, str(out / "model.safetensors"))
    print(f"[3/3] safetensors: {len(sd)} tensors -> {out / 'model.safetensors'}", flush=True)

    cfg_d = cfg.to_dict()
    with open(out / "config.json", "w") as f:
        json.dump(cfg_d, f, indent=2, default=str)
    print(f"      config.json -> {out / 'config.json'}", flush=True)

    print("\nDone. Smoke-load with:")
    print(
        f"  python3 -c \"import sglang as sgl; "
        f"e = sgl.Engine(model_path='{out}', skip_tokenizer_init=True, "
        f"tp_size=8, dtype='bfloat16', attention_backend='flashinfer', "
        f"linear_attn_backend='triton'); "
        f"print(e.generate({{'input_ids':[[1,2,3,4]]}}, "
        f"{{'max_new_tokens':4}}))\""
    )


if __name__ == "__main__":
    main()
