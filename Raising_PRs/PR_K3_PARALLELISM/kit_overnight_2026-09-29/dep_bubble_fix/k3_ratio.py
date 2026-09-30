"""K3 text : vision training compute per step, from the K3 report (104.2B activated, 24 MLA of 93 layers, 96 heads,
MoonViT-V2 0.4B / 27 layers / patch 14, 2x2 pixel shuffle) and K2.5 (4-frame chunks, 4x temporal pooling after the
tower). Assumed: MLA score/value head dims 192/128 (K2-style), ViT width 1152 (401M over 27 layers), spatial attention
within a frame, the temporal pass and KDA negligible; LLM trains at 3x its forward, the DEP tower at 4x (forward,
recompute, backward)."""
L_ACT = 104.2e9
def llm_fwd(ctx):  # FLOPs per token
    return 2 * L_ACT + 24 * 2 * (ctx / 2) * 96 * (192 + 128)
def vit_fwd_patch(w, h):
    n = (w // 14) * (h // 14)
    return 2 * 0.401e9 + 27 * 4 * n * 1152
cases = [("448 px image", 448, 448, 4), ("3584 px image", 3584, 3584, 4), ("720p video", 1280, 720, 16), ("1080p video", 1920, 1080, 16)]
print("| input | ctx | ViT/LLM per vision token (training) | vision share at v = 10% / 20% / 50% / 90% of tokens |")
print("|---|---|---:|---|")
for name, w, h, patches in cases:
    for ctx in (8192, 65536):
        rho = 4 * patches * vit_fwd_patch(w, h) / (3 * llm_fwd(ctx))
        shares = [v * rho / (1 + v * rho) for v in (0.1, 0.2, 0.5, 0.9)]
        print(f"| {name} | {ctx // 1024}K | {rho:.3f} | " + " / ".join(f"{s:.1%}" for s in shares) + " |")
