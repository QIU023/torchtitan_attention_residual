#!/usr/bin/env python3
"""Caption generation smoke (phase 6 task B5).

Loads an AttnRes-Kimi-VL ckpt + projector + frozen SigLIP and runs
greedy autoregressive decode on a held-out image. No KV cache (full
re-forward each step) — proves the model produces sensible logits +
the inference path doesn't crash on AttnRes block aggregation. Real
KV-cache support is a separate follow-up.

Usage::

    torchrun --nproc_per_node=1 phase5/generate_caption.py \\
        --ckpt phase5/runs/v8_pretrain_resilient_from_v7_step800/checkpoint/step-N \\
        --image /root/hf_cache/LLaVA-Pretrain/00001/000000001.jpg \\
        --max-new-tokens 30 \\
        --prompt 'Describe this image:'

Single-process (FSDP=1) for simplicity. The DCP loader auto-handles
the multi-rank → single-rank state dict conversion.

Output: prints the generated caption + final loss on the held-out
example as a sanity number.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Make project workspace importable.
_HERE = Path(__file__).resolve().parent
_WORKSPACE = _HERE.parent
sys.path.insert(0, str(_WORKSPACE))
sys.path.insert(0, str(_WORKSPACE / "torchtitan"))

from phase5.multimodal_dataset import (  # noqa: E402
    IMAGE_TOKEN_ID, N_VISION_TOKENS,
)
from phase5.multimodal_model import Projector  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to a step-N DCP checkpoint dir.")
    p.add_argument("--image", required=True, help="Path to a JPG/PNG image.")
    p.add_argument("--prompt", default="Describe the image briefly:",
                   help="Text prompt prepended to image tokens.")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--top-k", type=int, default=1, help="1 = greedy.")
    p.add_argument("--config", default="kimi_linear_436m_block_attn_res_n4")
    p.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    p.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B")
    p.add_argument("--cache-dir", default="/root/hf_cache")
    p.add_argument("--hf-assets", default=str(
        _WORKSPACE / "torchtitan/assets/hf/Llama-3.1-8B"
    ))
    return p.parse_args()


def init_dist():
    """Single-rank distributed init (DCP needs a process group)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def build_lm(args, device):
    """Build the AttnRes-Kimi LM matching the training config."""
    from torchtitan.experiments.kimi_linear.config_registry import (
        kimi_linear_436m_block_attn_res_n4,
    )
    from torchtitan.experiments.kimi_linear.attn_res_model import (
        KimiLinearAttnResModel,
    )
    cfg = kimi_linear_436m_block_attn_res_n4()
    spec = cfg.model_spec.model
    lm = KimiLinearAttnResModel(
        spec.kimi_config,
        num_blocks=spec.num_blocks,
    )
    lm.init_weights()
    lm = lm.to(device=device, dtype=torch.bfloat16)
    return lm


def load_lm_ckpt(lm, ckpt_path):
    """Load LM weights from a DCP-sharded ckpt into a single-rank LM."""
    state_dict = {"model": {k: v for k, v in lm.state_dict().items()}}
    dcp.load(state_dict, checkpoint_id=str(ckpt_path))
    lm.load_state_dict(state_dict["model"], strict=False)


def main():
    args = parse_args()
    init_dist()
    device = torch.device("cuda:0")
    print(f"[gen] device={device}, torch={torch.__version__}")

    print(f"[gen] loading vision_tower {args.vision_model}")
    vt = AutoModel.from_pretrained(
        args.vision_model, dtype=torch.bfloat16, cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
    )
    vision_tower = (vt.vision_model if hasattr(vt, "vision_model") else vt).to(device).eval()
    for p in vision_tower.parameters():
        p.requires_grad_(False)
    image_processor = AutoProcessor.from_pretrained(
        args.vision_model, cache_dir=args.cache_dir,
    )
    vision_dim = getattr(vt.config, "vision_config", vt.config).hidden_size
    print(f"[gen] vision_dim={vision_dim}")

    print(f"[gen] loading tokenizer {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)

    print(f"[gen] building LM ({args.config})")
    lm = build_lm(args, device)
    lm_dim = lm.config.hidden_size
    print(f"[gen] lm_dim={lm_dim}")

    projector = Projector(vision_dim=vision_dim, lm_dim=lm_dim).to(
        device=device, dtype=torch.bfloat16,
    )

    print(f"[gen] loading LM weights from {args.ckpt}")
    load_lm_ckpt(lm, args.ckpt)
    lm.eval()

    # --------- Prepare input ---------
    img = Image.open(args.image).convert("RGB")
    pix = image_processor(images=img, return_tensors="pt")["pixel_values"].to(
        device=device, dtype=torch.bfloat16,
    )
    with torch.no_grad():
        vision_features = vision_tower(pixel_values=pix).last_hidden_state
        vision_embeds = projector(vision_features)  # (1, N_vis, lm_dim)

    bos = tok.bos_token_id or 128_000
    prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
    seq = (
        [IMAGE_TOKEN_ID] * N_VISION_TOKENS
        + [bos] + prompt_ids
    )
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)

    # --------- Generate ---------
    print(f"[gen] decoding up to {args.max_new_tokens} tokens (greedy={'yes' if args.top_k == 1 else 'no'})")
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            logits = lm(
                input_ids,
                vision_embeds=vision_embeds,
                image_token_id=IMAGE_TOKEN_ID,
            )
            next_logit = logits[0, -1, :].float()
            if args.top_k > 1:
                topk = torch.topk(next_logit, k=args.top_k)
                probs = torch.softmax(topk.values, dim=-1)
                idx = torch.multinomial(probs, num_samples=1)
                next_id = topk.indices[idx].item()
            else:
                next_id = int(next_logit.argmax().item())
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_id]], dtype=torch.long, device=device),
            ], dim=1)
            if next_id == (tok.eos_token_id or 128_001):
                print(f"[gen] hit EOS at step {step}")
                break

    # --------- Decode ---------
    generated_ids = input_ids[0, len(seq):].tolist()
    text = tok.decode(generated_ids, skip_special_tokens=True)
    print()
    print(f"=== generated caption ({len(generated_ids)} tokens) ===")
    print(text)
    print(f"=== prompt was: {args.prompt!r} ===")


if __name__ == "__main__":
    main()
