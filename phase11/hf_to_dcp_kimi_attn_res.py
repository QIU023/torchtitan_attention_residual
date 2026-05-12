"""HF safetensors → DCP conversion for the Kimi Linear AttnRes LM.

Reverse of ``phase10/dcp_to_hf_kimi_attn_res.py``. Builds an empty
torchtitan-format Kimi AttnRes model, materialises its parameter
shapes, then fills each from the corresponding HF safetensors keys
using the inverse of ``remap_one``. Writes the resulting state dict
to a DCP directory loadable by torchtitan's CheckpointManager.

The MoE expert weights are the only non-trivial inverse: HF stores
each expert separately (``mlp.experts.{e}.gate_proj.weight`` etc.);
the tt-format DCP wants them stacked (``ffn._moe.experts.w1`` of
shape ``[E, intermediate, hidden]``).

Usage::

    PYTHONPATH=torchtitan:. python phase11/hf_to_dcp_kimi_attn_res.py \\
        --in /root/torchtitan_attention_residual/phase11/hf/vlm_sft_3ep \\
        --out /workspace/torchtitan_attention_residual/phase5/runs/vlm_447m_sft_3ep/checkpoint/step-0 \\
        --config kimi_linear_447m_aligned_block_attn_res_n4

Vision tower and projector weights present in the HF dir are SKIPPED
(only LM goes into the DCP — the trainer doesn't update vision; the
SGLang generator loads vision from the same HF dir at startup).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open


_W_TO_HF = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_HF_TO_W = {v: k for k, v in _W_TO_HF.items()}


def build_skeleton(config_name: str):
    """Construct empty KimiLinearAttnResModel matching config_name.

    Returns (kimi_config, model). Model weights initialised but values
    don't matter — we'll overwrite them all from HF.
    """
    from torchtitan.experiments.kimi_linear import config_registry
    from torchtitan.experiments.kimi_linear.attn_res_model import (
        KimiLinearAttnResModel,
    )

    builder = getattr(config_registry, config_name, None)
    if builder is None:
        raise NotImplementedError(
            f"Unknown config '{config_name}'. Available: "
            + ", ".join(
                n for n in dir(config_registry) if n.startswith("kimi_linear_")
            )
        )
    cfg = builder()
    spec = cfg.model_spec.model
    model = KimiLinearAttnResModel(spec.kimi_config, num_blocks=spec.num_blocks)
    model.init_weights()
    return spec.kimi_config, model


def load_hf_safetensors(in_dir: Path) -> dict[str, torch.Tensor]:
    """Read all .safetensors files in in_dir into a flat dict."""
    out: dict[str, torch.Tensor] = {}
    files = sorted(in_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {in_dir}")
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as handle:
            for k in handle.keys():
                out[k] = handle.get_tensor(k)
    return out


def inv_remap(
    tt_key: str,
    hf_state: dict[str, torch.Tensor],
    num_experts: int,
) -> torch.Tensor | None:
    """Map one tt state_dict key to its HF source tensor(s).

    Returns the assembled tensor for ``tt_key``, or ``None`` if no
    matching HF source (e.g. mm_projector key in LM-only converter).
    """

    # --- Top-level singletons --- #
    direct_map = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
        "final_attn_res_proj.weight": "model.final_attn_res_proj.weight",
        "final_attn_res_norm.weight": "model.final_attn_res_norm.weight",
    }
    if tt_key in direct_map:
        hf_key = direct_map[tt_key]
        t = hf_state.get(hf_key)
        # tied embedding case: if lm_head is missing in HF, reuse embed
        if t is None and tt_key == "lm_head.weight":
            return hf_state.get("model.embed_tokens.weight")
        return t

    # --- Per-layer --- #
    if not tt_key.startswith("layers."):
        return None
    rest = tt_key[len("layers.") :]
    idx_s, _, sub = rest.partition(".")
    hf_prefix = f"model.layers.{idx_s}"

    # AttnRes weights pass through
    for tag in ("attn_res_proj", "attn_res_norm", "mlp_res_proj", "mlp_res_norm"):
        if sub == f"{tag}.weight":
            return hf_state.get(f"{hf_prefix}.{tag}.weight")

    # Layernorms
    if sub in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return hf_state.get(f"{hf_prefix}.{sub}")

    # self_attn.* — names mostly match; reshape KDA A_log [1,1,H,1] -> [H]
    if sub.startswith("self_attn."):
        t = hf_state.get(f"{hf_prefix}.{sub}")
        if t is not None and sub == "self_attn.A_log" and t.dim() == 4:
            t = t.reshape(-1)
        return t

    # Dense MLP (layer 0): ffn.{gate_proj,up_proj,down_proj}.weight
    # HF naming in our SFT dir uses 'mlp' prefix.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        if sub == f"ffn.{proj}.weight":
            return hf_state.get(f"{hf_prefix}.mlp.{proj}.weight")

    # MoE router/bias — yesterday's HF uses 'block_sparse_moe' prefix.
    # phase10 LM converter used 'mlp' — try both.
    if sub == "ffn._moe.router.gate.weight":
        for hf_key in (
            f"{hf_prefix}.block_sparse_moe.gate.weight",
            f"{hf_prefix}.mlp.gate.weight",
        ):
            t = hf_state.get(hf_key)
            if t is not None:
                return t
        return None

    if sub == "ffn._moe.expert_bias":
        for hf_key in (
            f"{hf_prefix}.block_sparse_moe.gate.e_score_correction_bias",
            f"{hf_prefix}.mlp.gate.e_score_correction_bias",
        ):
            t = hf_state.get(hf_key)
            if t is not None:
                return t
        return None

    # Fused MoE experts: stack HF per-expert tensors.
    # Yesterday's HF: 'block_sparse_moe.experts.{e}.w{1,2,3}.weight'
    # phase10 HF:     'mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight'
    if sub.startswith("ffn._moe.experts."):
        w_tag = sub[len("ffn._moe.experts.") :]  # "w1" | "w2" | "w3"
        hf_tag = _W_TO_HF.get(w_tag)
        if hf_tag is None:
            return None
        per_expert = []
        for e in range(num_experts):
            t = hf_state.get(
                f"{hf_prefix}.block_sparse_moe.experts.{e}.{w_tag}.weight"
            )
            if t is None:
                t = hf_state.get(
                    f"{hf_prefix}.mlp.experts.{e}.{hf_tag}.weight"
                )
            if t is None:
                return None
            per_expert.append(t)
        return torch.stack(per_expert, dim=0).contiguous()

    # Shared experts.
    # Yesterday's HF: 'block_sparse_moe.shared_experts.{gate_proj,up_proj,down_proj}.weight'
    # phase10 HF:     'mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight'
    if sub.startswith("ffn._moe.shared_experts."):
        tail = sub[len("ffn._moe.shared_experts.") :]
        w_tag, _, suff = tail.partition(".")  # "w1" + "weight"
        hf_tag = _W_TO_HF.get(w_tag)
        if hf_tag is None:
            return None
        for hf_key in (
            f"{hf_prefix}.block_sparse_moe.shared_experts.{hf_tag}.{suff}",
            f"{hf_prefix}.mlp.shared_experts.{hf_tag}.{suff}",
        ):
            t = hf_state.get(hf_key)
            if t is not None:
                return t
        return None

    return None


def init_dist_singleton():
    """Init a single-rank process group for dcp.save."""
    if dist.is_initialized():
        return
    import os
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, type=Path)
    ap.add_argument("--out", dest="out_dir", required=True, type=Path)
    ap.add_argument(
        "--config", required=True,
        help="torchtitan flavor name (must match the trainer's --flavor)",
    )
    args = ap.parse_args()

    print(f"[hf->dcp] loading HF from {args.in_dir}")
    hf_state = load_hf_safetensors(args.in_dir)
    print(f"[hf->dcp] HF keys: {len(hf_state)}")

    print(f"[hf->dcp] building skeleton with config={args.config}")
    kimi_config, model = build_skeleton(args.config)
    print(f"[hf->dcp]   num_experts={kimi_config.num_experts}")
    print(f"[hf->dcp]   num_hidden_layers={kimi_config.num_hidden_layers}")

    tt_state = model.state_dict()
    print(f"[hf->dcp] tt state_dict keys: {len(tt_state)}")

    filled = 0
    unmapped = []
    for tt_key, dst_tensor in tt_state.items():
        src = inv_remap(tt_key, hf_state, kimi_config.num_experts)
        if src is None:
            unmapped.append(tt_key)
            continue
        if src.shape != dst_tensor.shape:
            print(
                f"[hf->dcp] SHAPE MISMATCH for {tt_key}: "
                f"src={tuple(src.shape)} dst={tuple(dst_tensor.shape)}"
            )
            unmapped.append(tt_key)
            continue
        dst_tensor.copy_(src.to(dtype=dst_tensor.dtype))
        filled += 1

    print(f"[hf->dcp] filled {filled}/{len(tt_state)} keys")
    if unmapped:
        print(f"[hf->dcp] UNMAPPED ({len(unmapped)}): {unmapped[:5]}...")
        # Bail if anything important is missing
        critical = [k for k in unmapped if "embed" in k or "lm_head" in k]
        if critical:
            raise RuntimeError(f"Critical keys unmapped: {critical}")

    # dcp.save needs a ModelWrapper-style state_dict path.
    # Use torchtitan's CheckpointManager wrapper conventions.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    init_dist_singleton()
    print(f"[hf->dcp] writing DCP to {args.out_dir}")
    # Save FLAT state_dict (no "model" wrapper). torchtitan's PolicyTrainer
    # / dcp.load expects keys like "embed_tokens.weight" at top level, not
    # "model.embed_tokens.weight". An earlier wrapping {"model": ...} caused
    # RuntimeError "Missing key in checkpoint state_dict: embed_tokens.weight".
    save_state = dict(model.state_dict())
    dcp.save(save_state, checkpoint_id=str(args.out_dir))
    print(f"[hf->dcp] DONE — {filled} keys written")

    # Manifest for traceability
    manifest = {
        "source_hf": str(args.in_dir),
        "config": args.config,
        "num_experts": kimi_config.num_experts,
        "num_hidden_layers": kimi_config.num_hidden_layers,
        "filled_keys": filled,
        "unmapped_keys": unmapped,
    }
    with (args.out_dir / "conversion_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
