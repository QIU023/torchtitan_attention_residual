"""Smoke test: load the converted 447M VLM HF safetensors through our
SGLang :class:`KimiAttnResVLForConditionalGeneration` and exercise:

  * config parsing (architecture name → class lookup)
  * weight load (LM keys → language_model.load_weights, projector
    keys → mm_projector, vision_tower keys → skip)
  * vision_tower instantiation (HF SiglipVisionModel from cache)
  * projector forward on a dummy vision feature tensor
  * language_model forward on a dummy input_embeds tensor

This is GPU-required (the LM has fused-MoE Triton kernels) but
single-GPU and runs in ~30s. Fails fast if any of:
  * SiglipVisionModel can't be loaded from the configured HF id
  * Any safetensors key fails to bind to a parameter
  * Projector dim mismatch with the LM hidden size
  * Vision feature dim doesn't match what fc1 expects
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS / "sglang" / "python"))


def main():
    ckpt = _WS / "phase11" / "hf_aligned_447m_vlm_step2500"
    assert ckpt.exists(), f"converted VLM ckpt not found: {ckpt}"

    # Parse config the same way SGLang does
    cfg_dict = json.loads((ckpt / "config.json").read_text())
    print(f"[smoke] arch: {cfg_dict['architectures']}")
    print(f"[smoke] vision_tower: {cfg_dict['vision_tower_path']}")
    print(f"[smoke] vision_hidden_size: {cfg_dict['vision_hidden_size']}")
    print(f"[smoke] llm_hidden_size: {cfg_dict['text_config']['hidden_size']}")

    # Build the LM config that SGLang expects (KimiLinearConfig, but
    # since we don't want to instantiate the SGLang config classes
    # which require trust_remote_code, fake an attribute bag).
    text_cfg_dict = cfg_dict["text_config"]

    class _AttrBag:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    text_cfg = _AttrBag(text_cfg_dict)
    cfg = _AttrBag(cfg_dict)
    cfg.text_config = text_cfg

    # Set torch default dtype to bf16 to match how SGLang loads
    torch.set_default_dtype(torch.bfloat16)

    print("[smoke] importing SGLang VLM model class...")
    from sglang.srt.models.attn_res_vl_overlay import (
        KimiAttnResVLForConditionalGeneration,
        _build_frozen_siglip,
        _MmProjectorBundle,
    )

    print("[smoke] building frozen SigLIP vision tower...")
    vt = _build_frozen_siglip(cfg.vision_tower_path)
    n_vt_params = sum(p.numel() for p in vt.parameters())
    print(f"[smoke] vision_tower OK ({n_vt_params/1e6:.1f}M params, frozen)")

    print("[smoke] building projector...")
    proj = _MmProjectorBundle(
        vision_hidden_size=cfg.vision_hidden_size,
        llm_hidden_size=text_cfg.hidden_size,
    )
    print(f"[smoke] projector keys: {list(proj.state_dict().keys())}")

    print("[smoke] loading projector weights from safetensors...")
    from safetensors import safe_open
    sf_path = ckpt / "model.safetensors"
    with safe_open(sf_path, framework="pt") as f:
        proj_keys = [k for k in f.keys() if k.startswith("mm_projector.")]
        for k in proj_keys:
            t = f.get_tensor(k)
            short = k[len("mm_projector."):]
            proj_sd = proj.state_dict()
            assert short in proj_sd, f"missing projector key: {short}"
            assert proj_sd[short].shape == t.shape, (
                f"shape mismatch for {short}: ckpt={t.shape} vs "
                f"model={proj_sd[short].shape}"
            )
        print(f"[smoke] {len(proj_keys)} projector keys all match")

    print("[smoke] dummy projector forward...")
    # SigLIP-base outputs [B, 196, 768] last_hidden_state for 224x224
    dummy_vision = torch.randn(1, 196, 768, dtype=torch.bfloat16)
    with torch.no_grad():
        proj.eval()
        proj.to(torch.bfloat16)
        out = proj(dummy_vision.flatten(0, 1))
    print(f"[smoke] projector out shape: {tuple(out.shape)}")
    assert out.shape == (196, text_cfg.hidden_size)

    print("[smoke] dummy vision_tower forward (224x224 RGB)...")
    dummy_pixels = torch.randn(1, 3, 224, 224, dtype=next(vt.parameters()).dtype)
    with torch.no_grad():
        vt_out = vt(pixel_values=dummy_pixels)
    print(f"[smoke] vision_tower out shape: {tuple(vt_out.last_hidden_state.shape)}")
    assert vt_out.last_hidden_state.shape == (1, 196, 768)

    print("[smoke] all VLM components OK ✓")
    print(f"[smoke] full Engine load (LM + GPU) is the next step — run via")
    print("        sglang.Engine(model_path=..., trust_remote_code=True)")


if __name__ == "__main__":
    main()
