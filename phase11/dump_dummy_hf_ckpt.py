"""Build a random-init HF checkpoint matching the **phase4 436M** AttnRes
Kimi Linear shape, for SGLang smoke-validation.

The shape mirrors what ``phase4/launch_paperhparams_break3.sh`` actually
trained (config name ``kimi_linear_436m_block_attn_res_n4``):

* ``n_layers=16`` — paper Table 2, 436M row
* ``d_model=1168``, ``num_heads=16``, ``d_ff=528``
* MLA on layers 4/8/12/16 (1-indexed); KDA on the rest (3:1 ratio)
* MoE: 32 experts, top-8, 1 shared, ``moe_intermediate_size=528``,
  ``first_k_dense_replace=1`` (layer 0 dense)
* AttnRes ``num_blocks=4`` ⇒ ``layers_per_block=4``

When the real torchtitan DCP ckpt is converted via
``phase10/dcp_to_hf_kimi_attn_res.py`` and uploaded, swap the
``model.safetensors`` here for the converted one — the shapes and
config will match.

Output layout matches what ``sglang.Engine(model_path=...)`` expects::

    out/
      config.json          # architectures: [KimiBlockAttnResForCausalLM]
      model.safetensors    # all named_parameters as bf16

Run::

    cd /root/torchtitan_attention_residual && \\
    python3 phase11/dump_dummy_hf_ckpt.py --out phase11/hf_436m_random
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _bootstrap_single_proc():
    """Init torch.distributed + sglang model-parallel for a 1-proc dump.

    Uses gloo backend (CPU) so no GPU is touched during weight dump —
    important because the box may already be hosting an SGLang server
    that owns all GPU memory.
    """
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=os.environ.get("MASTER_PORT", "29557"),
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
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )

    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def _make_436m_config():
    """436M phase4 AttnRes structure (config_registry.SCALING_LAW_TABLE row "436m").

    Mirrors ``build_kimi_linear_config("436m")`` from torchtitan (see
    ``torchtitan/torchtitan/experiments/kimi_linear/config_registry.py``)
    translated to SGLang's nested ``linear_attn_config`` schema.
    """
    from sglang.srt.configs.kimi_linear import KimiLinearConfig

    # Paper Table 2, 436M row
    n_layers = 16
    d = 1168
    H = 16
    d_ff = 528

    # Head dim derivation from build_kimi_linear_config
    head_dim_mla_nope = max(32, d // H)             # 73
    head_dim_mla_rope = max(16, head_dim_mla_nope // 2)  # 36
    head_dim_mla_v = head_dim_mla_nope              # 73
    kda_head_dim = head_dim_mla_nope                # 73
    kv_lora_rank = d // 2                           # 584

    # KDA:MLA = 3:1 → MLA on every 4th layer (1-indexed)
    period = 4
    kda_layers = [i for i in range(1, n_layers + 1) if i % period != 0]
    full_attn_layers = [i for i in range(1, n_layers + 1) if i % period == 0]

    cfg = KimiLinearConfig(
        # Vocabulary / embedding
        vocab_size=163840,
        hidden_size=d,
        tie_word_embeddings=True,
        # Depth / width
        num_hidden_layers=n_layers,
        intermediate_size=d_ff,
        num_attention_heads=H,
        num_key_value_heads=H,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        # MLA
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=head_dim_mla_nope,
        qk_rope_head_dim=head_dim_mla_rope,
        v_head_dim=head_dim_mla_v,
        mla_use_nope=True,
        # KDA (nested in linear_attn_config for SGLang)
        linear_attn_config={
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            "num_heads": H,
            "head_dim": kda_head_dim,
            "short_conv_kernel_size": 4,
        },
        # MoE
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
        # AttnRes (consumed via getattr in kimi_block_attn_res.py)
        attn_res_num_blocks=4,
        # Two-arch list: SGLang's ModelRegistry walks the list in order
        # and resolves the first match (our AttnRes class). The MLA
        # detection in ``model_config.py:498`` requires
        # ``KimiLinearForCausalLM`` to appear *somewhere* in the list to
        # set ``attention_arch = AttentionArch.MLA`` — without that,
        # the engine picks the non-MLA flashinfer backend which doesn't
        # support the ``q_rope`` kwarg the MLA forward path passes.
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
    """Construct model and ensure each param has random non-zero data.

    Crucially, **do not** cast the model to a single dtype: the SGLang
    KimiDeltaAttention layer pins ``A_log``, ``dt_bias``, and the
    ``qkv_conv1d`` weights at fp32 in its __init__ (via
    ``params_dtype=torch.float32``); the model also pre-quantizes some
    paths' bias to fp32 for numerics. A blanket
    ``model.to(dtype=bf16)`` would cast those back to bf16, which then
    makes cuda-graph capture fail with "input and bias should have the
    same dtype" because the runtime input flowing in is already bf16
    while the bias is the (now-bf16-but-supposed-to-be-fp32) param.

    Save each param in its natural dtype; SGLang's loader copies into
    the live param via ``Tensor.copy_``, which auto-promotes/demotes.
    """
    from sglang.srt.models.kimi_block_attn_res import KimiBlockAttnResForCausalLM
    with torch.device("cpu"):
        m = KimiBlockAttnResForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.numel() == 0 or not p.dtype.is_floating_point:
                continue
            # Leave attn_res pseudo-queries at zero (paper § 5: zero-init
            # so the initial softmax is uniform and Block AttnRes
            # degenerates to a vanilla residual stream).
            if "attn_res_proj" in n or "final_attn_res_proj" in n:
                continue
            if p.abs().max().item() == 0:
                p.normal_(mean=0.0, std=0.02)
    return m


# ---------------------------------------------------------------------------
# Live → HF unfused state_dict translation
# ---------------------------------------------------------------------------
# The live SGLang model holds:
#   • ``self_attn.fused_qkvbfg_a_proj.weight`` packing q/k/v/b/f_a/g_a
#   • ``self_attn.fused_fg_b_proj.weight`` packing f_b/g_b (3D batch)
#   • ``self_attn.qkv_conv1d.weight`` packing q_conv1d/k_conv1d/v_conv1d
#     (with an extra unit dim from the model's __init__ unsqueeze(1))
#   • ``mlp.experts.w13_weight`` and ``w2_weight`` (stacked over experts)
#   • ``mlp.shared_experts.gate_up_proj.weight`` packing gate+up
#   • Layer-0 dense ``mlp.gate_up_proj.weight`` packing gate+up
# load_weights expects each of those split out under HF/Kimi names.

def _live_to_hf_state_dict(model, cfg) -> dict:
    """Walk live state_dict, emit HF unfused naming.

    Mirrors what ``phase10/dcp_to_hf_kimi_attn_res.py`` produces from a
    torchtitan DCP, so the dummy and the real ckpt are loadable through
    the same code path.
    """
    raw = model.state_dict()
    out: dict[str, torch.Tensor] = {}

    seen_storages: set[int] = set()  # dedupe aliased storages

    def emit(name: str, tensor: torch.Tensor):
        if tensor.numel() == 0:
            return
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in seen_storages:
            return
        seen_storages.add(ptr)
        out[name] = tensor.detach().cpu().contiguous()

    # KDA fused-projection sizes
    kda = cfg.linear_attn_config
    kda_num_heads = kda["num_heads"]
    kda_head_dim = kda["head_dim"]
    kda_proj = kda_head_dim * kda_num_heads  # projection_size in KimiDeltaAttention

    moe_inter = cfg.moe_intermediate_size

    # In KimiDecoderLayer, MoE layers register both ``block_sparse_moe`` and
    # ``mlp`` pointing at the same KimiMoE instance; PyTorch's
    # ``named_parameters`` dedup picks the FIRST registration, which is
    # ``block_sparse_moe.*``. SGLang's load_weights then looks the param up
    # under the same prefix, so the checkpoint MUST also use
    # ``block_sparse_moe.*`` for MoE layers. Dense layer 0 has only
    # ``mlp.*`` (no MoE registration), so it stays as-is.
    #
    # Drop the ``mlp.*`` alias when the block_sparse_moe sibling exists.
    # Drop ``self_attn.attn.{A_log,dt_bias}`` (RadixLinearAttention re-export
    # of the parent's params).
    def is_alias(k: str) -> bool:
        if ".mlp." in k:
            sibling = k.replace(".mlp.", ".block_sparse_moe.")
            if sibling in raw:
                return True
        if k.endswith(".self_attn.attn.A_log") or k.endswith(
            ".self_attn.attn.dt_bias"
        ):
            return True
        return False

    for k, v in raw.items():
        if v.ndim == 0:
            continue
        if is_alias(k):
            continue

        # ----- KDA fused projections -----------------------------------------
        if k.endswith(".self_attn.fused_qkvbfg_a_proj.weight"):
            # weight shape: [3*proj + num_heads + 2*head_dim, hidden]
            base = k.replace(".fused_qkvbfg_a_proj.weight", "")
            offsets = [
                ("q_proj.weight",   0,                                    kda_proj),
                ("k_proj.weight",   kda_proj,                             kda_proj),
                ("v_proj.weight",   2 * kda_proj,                         kda_proj),
                ("b_proj.weight",   3 * kda_proj,                         kda_num_heads),
                ("f_a_proj.weight", 3 * kda_proj + kda_num_heads,         kda_head_dim),
                ("g_a_proj.weight", 3 * kda_proj + kda_num_heads + kda_head_dim, kda_head_dim),
            ]
            for sub, start, size in offsets:
                emit(f"{base}.{sub}", v.narrow(0, start, size).clone())
            continue

        if k.endswith(".self_attn.fused_fg_b_proj.weight"):
            # weight shape: [2, projection_size, head_dim]
            base = k.replace(".fused_fg_b_proj.weight", "")
            emit(f"{base}.f_b_proj.weight", v[0].clone())
            emit(f"{base}.g_b_proj.weight", v[1].clone())
            continue

        if k.endswith(".self_attn.qkv_conv1d.weight"):
            # weight shape: [3*proj, 1, conv_size]  (model __init__ unsqueezes(1))
            # Keep the unit dim — MergedColumnParallelLinear.weight_loader
            # narrows the 3D param along dim 0 and asserts shape match, so
            # each per-shard checkpoint tensor must also be 3D
            # ([proj, 1, conv_size]).
            base = k.replace(".qkv_conv1d.weight", "")
            for sub, start in (("q_conv1d.weight", 0),
                               ("k_conv1d.weight", kda_proj),
                               ("v_conv1d.weight", 2 * kda_proj)):
                emit(f"{base}.{sub}", v.narrow(0, start, kda_proj).clone())
            continue

        # ----- MoE fused experts (routed) ------------------------------------
        # Live name: ``...block_sparse_moe.experts.w13_weight`` (canonical
        # after named_parameters dedup). Split per-expert under the same
        # block_sparse_moe.* prefix so SGLang's load_weights finds the
        # right entry in params_dict.
        if k.endswith(".block_sparse_moe.experts.w13_weight"):
            # shape: [E, 2*intermediate, hidden]; w1 = first half, w3 = second
            base = k.replace(".w13_weight", "")
            E = v.shape[0]
            for e in range(E):
                emit(f"{base}.{e}.w1.weight", v[e, :moe_inter, :].clone())
                emit(f"{base}.{e}.w3.weight", v[e, moe_inter:, :].clone())
            continue

        if k.endswith(".block_sparse_moe.experts.w2_weight"):
            base = k.replace(".w2_weight", "")
            E = v.shape[0]
            for e in range(E):
                emit(f"{base}.{e}.w2.weight", v[e].clone())
            continue

        # ----- Shared experts (gate_up packed; HF naming for the shards) -----
        if k.endswith(".block_sparse_moe.shared_experts.gate_up_proj.weight"):
            base = k.replace(".gate_up_proj.weight", "")
            half = v.shape[0] // 2
            emit(f"{base}.gate_proj.weight", v[:half].clone())
            emit(f"{base}.up_proj.weight",   v[half:].clone())
            continue

        # ----- Layer-0 dense MLP (no MoE; ``mlp.*`` only) --------------------
        if (
            k.endswith(".mlp.gate_up_proj.weight")
            and ".shared_experts." not in k
            and ".block_sparse_moe." not in k
        ):
            base = k.replace(".gate_up_proj.weight", "")
            half = v.shape[0] // 2
            emit(f"{base}.gate_proj.weight", v[:half].clone())
            emit(f"{base}.up_proj.weight",   v[half:].clone())
            continue

        # ----- Pass-through (everything else: layernorms, MLA projs,
        #       AttnRes pseudo-queries, embed, lm_head, norm, dt_bias, A_log)
        emit(k, v)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _bootstrap_single_proc()
    cfg = _make_436m_config()
    print(f"[1/3] config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")
    m = _build_model(cfg)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[2/3] model: {n_params:,} params")

    # ----- live → HF unfused state_dict ------------------------------------
    # SGLang's KimiLinearForCausalLM.load_weights() expects checkpoint
    # keys in the *unfused* convention (matches what
    # phase10/dcp_to_hf_kimi_attn_res.py emits and what Moonshot's HF
    # release uses). It then fuses them via stacked_params_mapping into
    # the live ``fused_*`` parameters. So: split each fused live tensor
    # back into the unfused HF names.
    sd = _live_to_hf_state_dict(m, cfg)

    from safetensors.torch import save_file
    save_file(sd, str(out / "model.safetensors"))
    print(f"[3/3] safetensors: {len(sd)} tensors -> {out/'model.safetensors'}")

    # Save HF-compatible config.json
    cfg_d = cfg.to_dict()
    with open(out / "config.json", "w") as f:
        json.dump(cfg_d, f, indent=2)
    print(f"          config.json -> {out/'config.json'}")

    # Minimal tokenizer placeholder (sglang Engine wants something parseable)
    # Use llama-style fallback: don't ship tokenizer, point engine at a known
    # tiny tokenizer. For now, write a stub that engine can detect as missing
    # and the user can override via --tokenizer-path.
    print("\nDone. Smoke-load with:")
    print(
        f"  python3 -c \"import sglang as sgl; "
        f"e = sgl.Engine(model_path='{out}', skip_tokenizer_init=True, "
        f"tp_size=1, dtype='bfloat16'); "
        f"print(e.generate({{'input_ids':[[1,2,3,4]]}}, "
        f"{{'max_new_tokens':4}}))\""
    )


if __name__ == "__main__":
    main()
