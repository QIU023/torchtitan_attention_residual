"""Modelled (推算) forward FLOPs of one vision encode per image against one text pipeline stage per micro-batch.

Models built on the meta device from the DEP tree's configs: the released Kimi-K3 flavor, and the debug model
widened to DEPW_DIM (tower as wide as the text, 2 layers). Forward FLOPs = 2 x the parameters a token (or patch)
touches, plus attention's quadratic terms: the tower attends fully within one image (4 P^2 qkv_dim per layer);
the text's MLA layers attend causally over the micro-batch (2 (T/2) heads (qk_head_dim + v_head_dim) per token);
KDA layers cost heads (4 d_k d_v + 2 x 64 (d_k + d_v)) per token. Routed experts count top_k / E of their
parameters. A text stage is the mean over the split's stages of the text layers plus the head.

Usage (CPU): CUDA_VISIBLE_DEVICES= PYTHONPATH=<kit>:. python flop_ratio.py   (cwd = the DEP tree)
"""

import os

import torch

RES_SIDES = {224: 224, 448: 448, 768: 756, 1024: 1008}
TOKENS = (1024, 2048, 4096, 8192)


def build(kind: str):
    if kind == "k3":
        from torchtitan.models.kimi_k3 import model_registry

        cfg = model_registry("Kimi-K3", enable_sp=False, seq_len=8192)
    else:
        os.environ["DEPW_DIM"] = kind
        import dep_ratio_local

        cfg = dep_ratio_local._widened(enable_sp=False, seq_len=8192)
    with torch.device("meta"):
        return cfg, cfg.build()


def text_layer_flops(cfg, layer_cfg, layer, tokens: int) -> float:
    frac = frac_of(layer_cfg) if getattr(layer_cfg, "moe", None) is not None else 1.0
    params = 0.0
    for name, p in layer.named_parameters():
        params += p.numel() * (frac if "routed_experts" in name else 1.0)
    flops = 2 * params * tokens
    if layer_cfg.attention is not None:
        a = layer_cfg.attention
        heads = a.n_heads
        qk = a.qk_nope_head_dim + a.qk_rope_head_dim
        flops += 2 * (tokens / 2) * heads * (qk + a.v_head_dim) * tokens
    else:
        d = layer_cfg.delta_attention
        heads, dk = d.num_heads, d.head_dim
        flops += heads * (4 * dk * dk + 2 * 64 * 2 * dk) * tokens
    return flops


def frac_of(layer_cfg) -> float:
    moe = layer_cfg.moe
    for obj in (moe, getattr(moe, "router", None), getattr(moe, "routed_experts", None)):
        if obj is None:
            continue
        k = getattr(obj, "top_k", None)
        e = getattr(obj, "num_experts", None)
        if k and e:
            return k / e
    raise RuntimeError("cannot find top_k / num_experts in the MoE config")


def tower_flops(cfg, model, side: int) -> float:
    enc = model.vision_encoder
    vcfg = cfg.vision_encoder
    patches = (side // 14) ** 2
    blk = next(iter(enc.layers.values())) if hasattr(enc.layers, "values") else enc.layers[0]
    layer_params = sum(p.numel() for p in blk.parameters())
    qkv = vcfg.block.attn.dim
    per_layer = 2 * layer_params * patches + 4 * patches * patches * qkv
    proj_params = sum(p.numel() for p in enc.projector.parameters())
    embed = 2 * enc.patch_embed.weight.numel() * patches
    return embed + vcfg.num_layers * per_layer + 2 * proj_params * patches / 4


def stage_flops(cfg, model, tokens: int, num_stages: int) -> float:
    total = 0.0
    layers = model.layers.values() if hasattr(model.layers, "values") else model.layers
    for lc, layer in zip(cfg.layers, layers):
        total += text_layer_flops(cfg, lc, layer, tokens)
    total += 2 * model.lm_head.weight.numel() * tokens
    return total / num_stages


def main():
    rows = []
    for kind, splits in (("k3", {"PP16 x VP2": 32, "pp2 x vpp4": 8}), (os.environ.get("RATIO_DIM", "6144"), {"pp2 x vpp4": 8})):
        cfg, model = build(kind)
        name = "Kimi-K3 (released)" if kind == "k3" else f"debug widened to dim {kind}"
        tw = {res: tower_flops(cfg, model, side) for res, side in RES_SIDES.items()}
        for split, n in splits.items():
            for t in TOKENS:
                st = stage_flops(cfg, model, t, n)
                rows.append((name, split, t, st, {res: tw[res] / st for res in RES_SIDES}))
        print(f"{name}: tower forward per image (TFLOP) " + ", ".join(f"{r}px {tw[r] / 1e12:.2f}" for r in RES_SIDES))
    print("\n| model | split | tokens per micro-batch | text stage forward (TFLOP) | one image 224 | 448 | 768 | 1024 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, split, t, st, r in rows:
        print(f"| {name} | {split} | {t} | {st / 1e12:.2f} | {r[224]:.3f} | {r[448]:.3f} | {r[768]:.3f} | {r[1024]:.3f} |")


if __name__ == "__main__":
    main()
