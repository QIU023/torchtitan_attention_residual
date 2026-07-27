"""Numerical parity against Kimi K3's OWN reference implementation.

Everything this repo knew about K3 before today came from the config and the
tech report, i.e. from inference. The HF model repo ships the reference modeling
code, so the architecture can be checked directly instead: instantiate the
official module, copy its weights into ours, feed identical inputs, compare.

That also produces the parameter-name mapping as a byproduct, which is what
loading official weights needs.

Run: PYTHONPATH=<titan> python ref_parity_probe.py [--verbose]
Requires CUDA (KDA and the MoE grouped GEMM are Triton/CUDA-only).
"""

from __future__ import annotations

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ref_loader import load as load_ref  # noqa: E402

VERBOSE = "--verbose" in sys.argv


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def ref_config(**over):
    """A small KimiLinearConfig for the reference, K3-shaped."""
    cfg_mod = load_ref("configuration_kimi_k3")
    base = dict(
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        q_lora_rank=64,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        mla_use_nope=True,
        mla_use_output_gate=True,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        rms_norm_eps=1e-5,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=2,
        moe_intermediate_size=128,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=True,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        moe_router_activation_func="sigmoid",
        first_k_dense_replace=1,
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 2,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
            # the reference config asserts on both, and its is_kda_layer uses
            # 1-based layer indices
            "kda_layers": [1],
            "full_attn_layers": [2, 3, 4],
        },
    )
    base.update(over)
    cfg = cfg_mod.KimiLinearConfig(**base)
    # The reference dispatches attention through
    # ALL_ATTENTION_FUNCTIONS[config._attn_implementation]; unset it is a
    # KeyError. eager keeps the comparison off flash-attention's own kernels.
    cfg._attn_implementation = "eager"
    return cfg


def our_config(ref_cfg):
    """The same structure expressed in our KimiLinearConfig."""
    from torchtitan.experiments.kimi_k3.model import KimiLinearConfig

    lac = ref_cfg.linear_attn_config
    return KimiLinearConfig(
        vocab_size=ref_cfg.vocab_size,
        hidden_size=ref_cfg.hidden_size,
        num_hidden_layers=ref_cfg.num_hidden_layers,
        intermediate_size=ref_cfg.intermediate_size,
        num_attention_heads=ref_cfg.num_attention_heads,
        num_key_value_heads=ref_cfg.num_key_value_heads,
        q_lora_rank=ref_cfg.q_lora_rank,
        kv_lora_rank=ref_cfg.kv_lora_rank,
        qk_nope_head_dim=ref_cfg.qk_nope_head_dim,
        qk_rope_head_dim=ref_cfg.qk_rope_head_dim,
        v_head_dim=ref_cfg.v_head_dim,
        mla_use_nope=True,
        mla_gated=True,
        attn_gate_param="full_rank",
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        rms_norm_eps=ref_cfg.rms_norm_eps,
        num_experts=ref_cfg.num_experts,
        num_experts_per_token=ref_cfg.num_experts_per_token,
        num_shared_experts=ref_cfg.num_shared_experts,
        moe_intermediate_size=ref_cfg.moe_intermediate_size,
        routed_expert_hidden_size=ref_cfg.routed_expert_hidden_size,
        latent_moe_use_norm=True,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        moe_router_activation_func="sigmoid",
        first_k_dense_replace=ref_cfg.first_k_dense_replace,
        kda_head_dim=lac["head_dim"],
        kda_num_heads=lac["num_heads"],
        kda_short_conv_kernel_size=lac["short_conv_kernel_size"],
        kda_gate_lower_bound=lac["gate_lower_bound"],
        kda_use_full_rank_gate=lac["use_full_rank_gate"],
        kda_layers=[1],
        full_attn_layers=[2, 3, 4],
        moe_layer_freq=1,
        num_expert_group=1,
        topk_group=1,
    )


def copy_matching(dst: torch.nn.Module, src: torch.nn.Module, mapping: dict):
    """Copy src params into dst using an explicit dst<-src name map."""
    sd_src = dict(src.named_parameters())
    sd_dst = dict(dst.named_parameters())
    missing = []
    with torch.no_grad():
        for d, s in mapping.items():
            if d not in sd_dst or s not in sd_src:
                missing.append((d, s, d in sd_dst, s in sd_src))
                continue
            if sd_dst[d].shape != sd_src[s].shape:
                missing.append((d, s, tuple(sd_dst[d].shape), tuple(sd_src[s].shape)))
                continue
            sd_dst[d].copy_(sd_src[s])
    return missing


# ---------- MLA ---------------------------------------------------------- #

_MLA_MAP = {
    "q_a_proj.weight": "q_a_proj.weight",
    "q_a_layernorm.weight": "q_a_layernorm.weight",
    "q_b_proj.weight": "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "kv_a_layernorm.weight",
    "kv_b_proj.weight": "kv_b_proj.weight",
    "o_proj.weight": "o_proj.weight",
    # OUR NAME IS DIFFERENT: official calls the Gated-MLA output gate g_proj.
    "attn_gate_proj.weight": "g_proj.weight",
}


def check_mla(device="cuda"):
    from torchtitan.experiments.kimi_k3.model import KimiMLAAttention

    ref_mod = load_ref("modeling_kimi_linear")
    rc = ref_config()
    torch.manual_seed(0)
    ref = ref_mod.KimiMLAAttention(rc, layer_idx=2).to(device).float()
    ours = KimiMLAAttention(our_config(rc), layer_idx=2).to(device).float()
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.05)
    bad = copy_matching(ours, ref, _MLA_MAP)

    B, T = 2, 16
    x = torch.randn(B, T, rc.hidden_size, device=device)
    # Our forward hardcodes a causal mask; hand the reference the same one so
    # the comparison is of the projections and the gate, not of masking policy.
    causal = torch.zeros(B, 1, T, T, device=device)
    causal.masked_fill_(
        torch.ones(T, T, dtype=torch.bool, device=device).triu(1),
        float("-inf"),
    )
    with torch.no_grad():
        ours_out = ours(x)
        ref_out = ref(x, attention_mask=causal)
    ref_out = ref_out[0] if isinstance(ref_out, tuple) else ref_out
    return {
        "unmapped": bad,
        "rel": rel(ours_out, ref_out),
        "ref_params": sorted(n for n, _ in ref.named_parameters()),
        "our_params": sorted(n for n, _ in ours.named_parameters()),
    }


# ---------- KDA ---------------------------------------------------------- #

_KDA_MAP = {
    f"{n}.weight": f"{n}.weight"
    for n in (
        "q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj",
        "b_proj", "g_proj", "o_proj",
    )
}
_KDA_MAP.update(
    {
        "q_conv1d.weight": "q_conv1d.weight",
        "k_conv1d.weight": "k_conv1d.weight",
        "v_conv1d.weight": "v_conv1d.weight",
        "o_norm.weight": "o_norm.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
    }
)


def check_kda(device="cuda"):
    from torchtitan.experiments.kimi_k3.model import KimiDeltaAttention

    ref_mod = load_ref("modeling_kimi_linear")
    rc = ref_config()
    torch.manual_seed(0)
    ref = ref_mod.KimiDeltaAttention(rc, layer_idx=1).to(device).to(torch.bfloat16)
    ours = KimiDeltaAttention(our_config(rc), layer_idx=1).to(device).to(
        torch.bfloat16
    )
    for p in ref.parameters():
        if p.dtype.is_floating_point and p.dim() > 0:
            torch.nn.init.normal_(p, std=0.05)
    bad = copy_matching(ours, ref, _KDA_MAP)

    # T > 64 so both take the chunk path
    x = torch.randn(1, 128, rc.hidden_size, device=device, dtype=torch.bfloat16)
    ours.train()
    ref.train()
    with torch.no_grad():
        ours_out = ours(x)
        ref_out = ref(x)
    ref_out = ref_out[0] if isinstance(ref_out, tuple) else ref_out
    return {
        "unmapped": bad,
        "rel": rel(ours_out, ref_out),
        "ref_params": sorted(n for n, _ in ref.named_parameters()),
        "our_params": sorted(n for n, _ in ours.named_parameters()),
    }


# ---------- MoE (Latent + SiTU + shared experts) ------------------------- #


def check_moe(device="cuda"):
    from torchtitan.experiments.kimi_k3.model import KimiMoE

    ref_mod = load_ref("modeling_kimi_linear")
    rc = ref_config()
    torch.manual_seed(0)
    ref = ref_mod.KimiSparseMoeBlock(rc).to(device).to(torch.bfloat16)
    ours = KimiMoE(our_config(rc)).to(device).to(torch.bfloat16)
    return {
        "ref_params": sorted(n for n, _ in ref.named_parameters()),
        "our_params": sorted(n for n, _ in ours.named_parameters()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    for name, fn in (("MLA", check_mla), ("KDA", check_kda), ("MoE", check_moe)):
        print(f"===== {name} =====", flush=True)
        try:
            r = fn()
        except Exception as e:  # keep going: one broken arm should not hide others
            print(f"  ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if "rel" in r:
            print(f"  rel vs reference: {r['rel']:.3e}", flush=True)
        if r.get("unmapped"):
            print(f"  UNMAPPED/SHAPE-MISMATCH: {r['unmapped']}", flush=True)
        ref_only = set(r["ref_params"]) - set(r["our_params"])
        our_only = set(r["our_params"]) - set(r["ref_params"])
        print(f"  in reference only: {sorted(ref_only)}", flush=True)
        print(f"  in ours only:      {sorted(our_only)}", flush=True)
        if VERBOSE:
            print(f"  ref params: {r['ref_params']}", flush=True)
            print(f"  our params: {r['our_params']}", flush=True)


if __name__ == "__main__":
    main()
