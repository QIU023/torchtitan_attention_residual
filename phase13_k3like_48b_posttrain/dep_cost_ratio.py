# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What is r, and what config change brings it into the range where bubbles can absorb it?

``r`` is one ViT forward in units of one text-stage forward. It is the only
model-dependent input to ``dep_hiding_theory.py``, and the hiding threshold from that
sweep is **r_eff <= 0.3** for the report's "most of the ViT computation is hidden".

FLOPs are estimated as ``2 * params * tokens`` plus the attention term that is quadratic
in sequence/patch count, and the parameter counts are MEASURED by constructing the modules
rather than derived from a hand-written formula -- MLA's lora compression and MoE's
activated-expert accounting are both easy to get wrong on paper, and the ratio is what
matters, so a consistent approximation on both sides is enough.

Usage:
    python3 dep_cost_ratio.py                       # current pp8vp4 flavor
    python3 dep_cost_ratio.py --seq 4096            # what a longer sequence does
    python3 dep_cost_ratio.py --scan                # candidate configs reaching r<=0.3
"""

from __future__ import annotations

import argparse


def _counts(kimi_config, vision_config):
    """Measured parameter counts: one text layer, and the whole vision tower."""
    import torch

    from torchtitan.models.kimi_k3.model import KimiK3Model
    from torchtitan.models.kimi_k3.moonvit import MoonViT

    with torch.device("meta"):
        tower = MoonViT(vision_config)
        text = KimiK3Model(kimi_config)

    proj = {id(p) for p in tower.mm_projector.parameters()}
    vit_params = sum(p.numel() for p in tower.parameters() if id(p) not in proj)
    proj_params = sum(p.numel() for p in tower.mm_projector.parameters())
    # ``layers`` is a ModuleDict of MIXED types (KDA and Gated MLA alternate), so a
    # stage's cost depends on which it holds. The mean is the right figure for a
    # ratio over stages, and taking layer 0 alone would bias it to whichever kind
    # happens to be first.
    layers = list(text.layers.values())
    per_text_layer = sum(p.numel() for m in layers for p in m.parameters()) / len(layers)
    return vit_params, proj_params, per_text_layer


def cost_ratio(
    *,
    vit_params: int,
    proj_params: int,
    per_text_layer: int,
    vit_layers: int,
    vit_hidden: int,
    vit_qkv_hidden: int,
    num_patches: int,
    merge: int,
    text_layers_per_stage: float,
    seq_len: int,
    text_attn_width: int,
) -> dict:
    """r = ViT forward cost / one text-stage forward cost, both in FLOPs."""
    # Linear (matmul) terms: 2 * params * tokens.
    vit_linear = 2 * vit_params * num_patches
    merged_tokens = num_patches // (merge * merge)
    vit_proj = 2 * proj_params * merged_tokens
    # Attention is quadratic in patch count: scores + weighted values, over all layers.
    vit_quad = 2 * 2 * vit_layers * num_patches * num_patches * vit_qkv_hidden

    text_linear = 2 * per_text_layer * text_layers_per_stage * seq_len
    text_quad = 2 * 2 * text_layers_per_stage * seq_len * seq_len * text_attn_width

    vit = vit_linear + vit_proj + vit_quad
    text = text_linear + text_quad
    return {
        "r": vit / text,
        "vit_linear": vit_linear,
        "vit_quad": vit_quad,
        "vit_quad_share": vit_quad / vit,
        "text_linear": text_linear,
        "text_quad": text_quad,
        "visual_token_share": merged_tokens / seq_len,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=None, help="override sequence length")
    ap.add_argument("--patches", type=int, default=None, help="override patches/image")
    ap.add_argument("--vit-hidden", type=int, default=None)
    ap.add_argument("--stages", type=int, default=32, help="total virtual stages")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    from torchtitan.models.kimi_k3.config_registry import (
        kimi_k3_debugmodel_report_arch_pp8vp4 as flavor,
    )

    cfg = flavor()
    kc = cfg.model_spec.model.kimi_config
    vc = cfg.model_spec.model.vision_config
    seq = args.seq if args.seq else cfg.training.seq_len
    patches = args.patches if args.patches else 1024  # collator's max_patches
    vit_params, proj_params, per_text_layer = _counts(kc, vc)
    merge = vc.merge_kernel_size[0]
    text_attn_width = kc.num_attention_heads * (kc.qk_nope_head_dim + kc.v_head_dim)

    base = dict(
        vit_params=vit_params,
        proj_params=proj_params,
        per_text_layer=per_text_layer,
        vit_layers=vc.num_hidden_layers,
        vit_hidden=vc.hidden_size,
        vit_qkv_hidden=vc.qkv_hidden_size,
        merge=merge,
        text_layers_per_stage=kc.num_hidden_layers / args.stages,
        text_attn_width=text_attn_width,
    )

    print(
        f"measured: vision tower {vit_params / 1e6:.2f}M + projector "
        f"{proj_params / 1e6:.2f}M, one text layer {per_text_layer / 1e6:.2f}M\n"
        f"text {kc.num_hidden_layers} layers over {args.stages} virtual stages = "
        f"{kc.num_hidden_layers / args.stages:.2f} layers/stage"
    )

    if not args.scan:
        res = cost_ratio(num_patches=patches, seq_len=seq, **base)
        print(
            f"\nseq_len {seq}, {patches} patches/image:\n"
            f"  r = {res['r']:.3f}   (threshold for 'most hidden' is 0.3)\n"
            f"  ViT quadratic share of ViT cost: {res['vit_quad_share'] * 100:.1f}%\n"
            f"  visual tokens as a share of the sequence: "
            f"{res['visual_token_share'] * 100:.1f}%"
        )
        return

    print(f"\n{'seq':>6} {'patches':>8} {'vis tok %':>10} {'r':>9}  {'verdict':<24}")
    for s in (256, 1024, 2048, 4096, 8192):
        for p in (1024, 512, 256):
            res = cost_ratio(num_patches=p, seq_len=s, **base)
            share = res["visual_token_share"] * 100
            if share > 100:
                verdict = "IMPOSSIBLE: tokens > seq"
            elif res["r"] <= 0.3:
                verdict = "OK: bubbles can absorb"
            elif res["r"] <= 1.0:
                verdict = "partial hiding only"
            else:
                verdict = "no hiding"
            print(f"{s:>6} {p:>8} {share:>9.1f}% {res['r']:>9.3f}  {verdict:<24}")
    print(
        "\n'vis tok %' is post-merge visual tokens against the sequence length. Above\n"
        "100% the config is not expressible -- one image cannot exceed the sequence it\n"
        "is spliced into, which is what the current 256-token sequence is up against."
    )


if __name__ == "__main__":
    main()
