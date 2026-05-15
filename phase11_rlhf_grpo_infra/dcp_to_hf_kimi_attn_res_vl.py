"""DCP → HF safetensors conversion for the **multimodal** Kimi AttnRes
ckpt (frozen SigLIP vision tower + 2-layer projector + AttnRes LM).

Counterpart to ``phase10_ckpt_dcp_to_hf/dcp_to_hf_kimi_attn_res.py`` (LM-only).
Differences:

1. The DCP state dict contains a top-level ``mm_projector.projector.*``
   block written by ``phase5_vlm_multimodal_sft/multimodal_model.py``'s ``Projector`` —
   we copy those weights through unchanged into the HF safetensors.

2. The vision tower (SigLIP) is **NOT** in the DCP — it was frozen
   throughout training. SGLang's :class:`KimiAttnResVLForConditionalGeneration`
   loads it fresh from the HF cache at __init__ time. So nothing to
   emit on the HF side either; we just record the path in config.json.

3. The HF ``config.json`` declares the multimodal architecture
   (``KimiAttnResVLForConditionalGeneration``) and embeds the LM
   config under ``text_config`` to match the SGLang loader pattern
   used by ``KimiVLForConditionalGeneration``.

Usage::

    torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
        --in phase5_vlm_multimodal_sft/runs/vlm_447m_pretrain/checkpoint/step-2500 \
        --out phase11_rlhf_grpo_infra/hf/vlm_pretrain \
        --config kimi_linear_447m_aligned_block_attn_res_n4 \
        --vision-tower google/siglip-base-patch16-224

The output dir is loadable via::

    sglang.Engine(model_path=..., trust_remote_code=True)

once the SGLang fork (with ``attn_res_vl_overlay.py``) is installed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))
sys.path.insert(0, str(_WS / "torchtitan"))

# Reuse the LM converter's helpers without importing it as a module
# (the existing script is a __main__ entry-point, not packaged).
_LM_CONV = _WS / "phase10" / "dcp_to_hf_kimi_attn_res.py"
_spec = importlib.util.spec_from_file_location("dcp_lm_conv", _LM_CONV)
_lm_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lm_conv)


# ---------------------------------------------------------------------------
# Multimodal skeleton
# ---------------------------------------------------------------------------

class _Projector(nn.Module):
    """Mirror of phase5_vlm_multimodal_sft/multimodal_model.py:Projector. Used only to
    materialise parameter slots so DCP load can populate them.
    """

    def __init__(self, vision_dim: int, lm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, lm_dim, bias=True)
        self.fc2 = nn.Linear(lm_dim, lm_dim, bias=True)


class _MmProjectorBundle(nn.Module):
    def __init__(self, vision_dim: int, lm_dim: int):
        super().__init__()
        self.projector = _Projector(vision_dim, lm_dim)


def _build_skeleton(config_name: str, vision_dim: int):
    """Build a mock VLM skeleton: LM (real model) + projector (real
    nn.Module). The vision tower is intentionally absent — it's not
    in DCP.
    """
    kimi_config, lm = _lm_conv.build_skeleton_state_dict(config_name)

    mm_projector = _MmProjectorBundle(
        vision_dim=vision_dim, lm_dim=kimi_config.hidden_size,
    )

    # Combine into a single state-dict-friendly bundle. Keys come out
    # like ``layers.0.self_attn.q_proj.weight`` (LM, no prefix) and
    # ``mm_projector.projector.fc1.weight`` (projector, with prefix).
    return kimi_config, lm, mm_projector


def make_hf_vlm_config(kimi_config, vision_tower_path: str, vision_dim: int) -> dict:
    """HF config.json content for the VLM. The text_config nested
    section is exactly what the LM-only converter emits.
    """
    text_cfg = _lm_conv.make_hf_config(kimi_config)
    # Strip architectures from the inner — only the outer wrapper
    # carries it.
    text_cfg.pop("architectures", None)

    return {
        # Both archs: the second triggers SGLang's MLA dispatch
        # (sets attention_arch=MLA so flashinfer_mla backend is
        # picked). Same workaround as the LM-only converter.
        "architectures": [
            "KimiAttnResVLForConditionalGeneration",
            "KimiLinearForCausalLM",
        ],
        "model_type": "kimi_attn_res_vl",
        "text_config": text_cfg,
        "vision_tower_path": vision_tower_path,
        "vision_hidden_size": vision_dim,
        "image_token_id": 32000,  # Llama-3.1 reserved; phase5 dataset
        "max_position_embeddings": text_cfg["max_position_embeddings"],
    }


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _remap_lm_state_dict(lm_sd, target_dtype):
    """Apply the same torchtitan→HF remap as the LM-only converter,
    but operate on a state dict instead of the whole-script flow.
    """
    _SHARED_W_TO_HF = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    out = {}
    for name, t in lm_sd.items():
        t = t.detach().to(target_dtype).contiguous()

        if name.startswith("layers."):
            i, rest = name[len("layers."):].split(".", 1)
            layer_prefix = f"model.layers.{i}"

            if (
                rest.startswith("ffn.gate_proj")
                or rest.startswith("ffn.up_proj")
                or rest.startswith("ffn.down_proj")
            ):
                out[f"{layer_prefix}.{rest.replace('ffn.', 'mlp.')}"] = t
                continue

            if rest in ("ffn._moe.experts.w1", "ffn._moe.experts.w2", "ffn._moe.experts.w3"):
                w_tag = rest.split(".")[-1]
                E = t.shape[0]
                for e in range(E):
                    out[
                        f"{layer_prefix}.block_sparse_moe.experts.{e}.{w_tag}.weight"
                    ] = t[e].contiguous()
                continue

            if rest == "ffn._moe.router.gate.weight":
                out[f"{layer_prefix}.block_sparse_moe.gate.weight"] = t
                continue

            if rest == "ffn._moe.expert_bias":
                out[f"{layer_prefix}.block_sparse_moe.gate.e_score_correction_bias"] = t
                continue

            if rest.startswith("ffn._moe.shared_experts."):
                tail = rest[len("ffn._moe.shared_experts."):]
                w_tag, _, suff = tail.partition(".")
                hf_tag = _SHARED_W_TO_HF[w_tag]
                out[
                    f"{layer_prefix}.block_sparse_moe.shared_experts.{hf_tag}.{suff}"
                ] = t
                continue

            if rest == "self_attn.A_log":
                out[f"{layer_prefix}.self_attn.A_log"] = t.reshape(1, 1, -1, 1)
                continue

            out[f"{layer_prefix}.{rest}"] = t
            continue

        if name == "lm_head.weight":
            out["lm_head.weight"] = t
            continue

        # Top-level non-layer keys (embed_tokens, norm, final_attn_res_*)
        out[f"model.{name}"] = t
    return out


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
    p.add_argument(
        "--config", default="kimi_linear_447m_aligned_block_attn_res_n4",
        help="torchtitan flavor name for the LM (used to materialise skeleton).",
    )
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--vision-tower", default="google/siglip-base-patch16-224",
        help="HF id or local path to the SigLIP vision tower. Recorded "
             "in config.json for runtime instantiation.",
    )
    p.add_argument(
        "--vision-hidden-size", type=int, default=768,
        help="Output dim of the vision tower's last_hidden_state. "
             "768 for siglip-base-patch16-224.",
    )
    p.add_argument(
        "--processor-source", type=Path, default=None,
        help="Optional path to a pre-existing HF VLM directory that "
             "already has processor configs (preprocessor_config.json, "
             "processor_config.json, tokenizer*, special_tokens_map.json). "
             "If provided, these 5 files are copied into --out-dir so "
             "SGLang's AutoProcessor.from_pretrained(out_dir) boots "
             "without manual intervention. Skipping this means SGLang "
             "engine boot will fail with 'cannot find processor in <out>' "
             "(known bug, 2026-05-11 overnight chain Stage C).",
    )
    args = p.parse_args()

    init_dist()
    torch.cuda.set_device(0)

    print(f"[conv-vl] building skeleton for config={args.config}")
    kimi_config, lm, mm_projector = _build_skeleton(
        args.config, vision_dim=args.vision_hidden_size,
    )

    # Build the COMBINED state dict. DCP keys are saved with the same
    # top-level layout the trainer constructed: LM keys flat at top
    # level + ``mm_state.projector.*`` for the trained projector.
    #
    # NOTE (2026-05-14): the projector now lives under ``mm_state.``
    # because phase5_vlm_multimodal_sft/train_mm.py registers it via ``_MMStateWrapper``
    # under the checkpointer key ``"mm_state"`` — its ``state_dict()``
    # emits ``{"projector": <model_sd>, "proj_optim": ..., "lm_optim":
    # ...}``, so DCP keys come out as ``mm_state.projector.fc1.weight``
    # etc. (The old ``mm_projector.projector.*`` prefix this converter
    # was written for predates that wrapper.) We only request the
    # ``mm_state.projector.*`` weights here — optimizer state
    # (``mm_state.proj_optim.*`` / ``mm_state.lm_optim.*``) is not
    # needed for an HF inference checkpoint, and DCP's partial-load
    # happily skips keys absent from the request dict.
    lm_sd = lm.state_dict()
    mm_sd = mm_projector.state_dict()  # keys: projector.fc1.weight, ...
    combined: dict[str, torch.Tensor] = {}
    for k, v in lm_sd.items():
        combined[k] = v
    for k, v in mm_sd.items():
        combined[f"mm_state.{k}"] = v

    print(
        f"[conv-vl] skeleton: {len(lm_sd)} LM keys + {len(mm_sd)} projector keys"
    )

    print(f"[conv-vl] loading DCP from {args.in_dir}")
    dcp.load(combined, checkpoint_id=str(args.in_dir))

    target_dtype = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
    }[args.dtype]

    # Split combined back into LM vs projector for separate handling.
    # Projector keys come back as ``mm_state.projector.fc1.weight`` etc.
    # (see the _MMStateWrapper note above). We rename them to the final
    # HF form ``mm_projector.projector.fc1.weight`` that the SGLang
    # attn_res_vl overlay loader expects.
    lm_state: dict[str, torch.Tensor] = {}
    projector_state: dict[str, torch.Tensor] = {}
    for k, v in combined.items():
        if k.startswith("mm_state.projector."):
            hf_k = "mm_projector." + k[len("mm_state."):]
            projector_state[hf_k] = v
        elif k.startswith("mm_state."):
            # optimizer state etc. — not requested, but be defensive.
            continue
        else:
            lm_state[k] = v

    # LM remap (same as phase10 LM-only converter).
    lm_hf_sd = _remap_lm_state_dict(lm_state, target_dtype)
    if (
        "model.embed_tokens.weight" in lm_hf_sd
        and "lm_head.weight" not in lm_hf_sd
    ):
        lm_hf_sd["lm_head.weight"] = lm_hf_sd["model.embed_tokens.weight"].clone()

    # Projector: keys are already in their final form
    # (``mm_projector.projector.fc1.weight`` etc.). Just dtype cast.
    proj_hf_sd = {
        k: v.detach().to(target_dtype).contiguous()
        for k, v in projector_state.items()
    }

    hf_sd = {**lm_hf_sd, **proj_hf_sd}

    print(
        f"[conv-vl] DCP keys: {len(combined)} → HF keys: {len(hf_sd)} "
        f"(LM: {len(lm_hf_sd)}, projector: {len(proj_hf_sd)})"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = make_hf_vlm_config(
        kimi_config,
        vision_tower_path=args.vision_tower,
        vision_dim=args.vision_hidden_size,
    )
    with (args.out_dir / "config.json").open("w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[conv-vl] wrote config.json")

    from safetensors.torch import save_file
    save_file(hf_sd, str(args.out_dir / "model.safetensors"))
    total_bytes = sum(t.element_size() * t.numel() for t in hf_sd.values())
    print(
        f"[conv-vl] wrote model.safetensors ({total_bytes / 1024**2:.1f} MB)"
    )

    # Auto-copy processor configs so SGLang's AutoProcessor.from_pretrained
    # boots without manual intervention. Failure to do this is a known
    # repeat-incident bug (overnight chain 2026-05-11 Stage C wasted ~30 min
    # to manually `cp` these from a prior working dir before GRPO could
    # launch). Make the failure mode impossible going forward.
    PROCESSOR_FILES = [
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    copied_processor_files: list[str] = []
    if args.processor_source is not None and args.processor_source.exists():
        import shutil
        for fname in PROCESSOR_FILES:
            src = args.processor_source / fname
            if src.exists():
                shutil.copy2(src, args.out_dir / fname)
                copied_processor_files.append(fname)
            else:
                print(f"[conv-vl] WARN: {src} missing in --processor-source")
        print(
            f"[conv-vl] copied {len(copied_processor_files)} processor files "
            f"from {args.processor_source}"
        )
    else:
        print(
            "[conv-vl] WARN: --processor-source not given; "
            "SGLang AutoProcessor.from_pretrained(out_dir) will fail. "
            "Pass --processor-source <prior_working_hf_dir> or copy the "
            f"5 files manually: {PROCESSOR_FILES}"
        )

    manifest = {
        "source_dcp": str(args.in_dir),
        "n_dcp_keys": len(combined),
        "n_hf_keys": len(hf_sd),
        "n_projector_keys": len(proj_hf_sd),
        "dtype": args.dtype,
        "total_bytes": total_bytes,
        "vision_tower": args.vision_tower,
        "processor_source": str(args.processor_source) if args.processor_source else None,
        "copied_processor_files": copied_processor_files,
        "key_sample": sorted(hf_sd.keys())[:20] + ["..."] + sorted(hf_sd.keys())[-5:],
    }
    with (args.out_dir / "conversion_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[conv-vl] DONE → {args.out_dir}")


if __name__ == "__main__":
    main()
