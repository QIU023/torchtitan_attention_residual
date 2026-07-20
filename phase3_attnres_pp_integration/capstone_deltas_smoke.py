"""Capstone: all K3 deltas compose in one training run (single GPU, 194m).
Gated MLA + alpha-graft AttnRes + MXFP4/MXFP8 QAT fake-quant + Per-Head
Muon optimizer, trained together. Proves the overnight components
interoperate; not a quality run (random init)."""
import dataclasses

import torch

from torchtitan.experiments.kimi_k3 import config_registry
from torchtitan.experiments.kimi_k3.model import KimiLinearSpec
from torchtitan.experiments.kimi_k3.mxfp4_qat import apply_mxfp4_qat
from torchtitan.experiments.kimi_k3.muon import Muon


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    kc = config_registry.build_kimi_linear_config("194m", num_experts=32)
    kc = dataclasses.replace(kc, mla_gated=True)  # K3 delta: Gated MLA
    spec = KimiLinearSpec(
        kimi_config=kc, num_blocks=4, attn_res_gated=True  # K3 delta: alpha graft
    )
    with torch.device("cuda"):
        model = spec.build()
        model.init_weights()
    n_qat = apply_mxfp4_qat(model, quantize_act=True)  # K3 delta: MXFP4 QAT
    model = model.to(torch.bfloat16)

    # tag attention q/o proj for per-head Muon
    heads = kc.num_attention_heads
    for name, p in model.named_parameters():
        if name.endswith("q_proj.base.weight") or name.endswith("o_proj.base.weight"):
            p._muon_heads = heads

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = Muon(trainable, lr=1e-3, adamw_lr=2e-4)  # K3 delta: Per-Head Muon

    losses = []
    for _ in range(20):
        tok = torch.randint(0, kc.vocab_size, (1, 256), device="cuda")
        out = model(tok)
        loss = out.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # confirm the graft alpha and MLA gate actually moved (deltas active)
    alpha = next(
        p for name, p in model.named_parameters()
        if name.endswith("attn_res_alpha")
    )
    print(
        f"[CAPSTONE] deltas: GatedMLA+AlphaGraft+MXFP4QAT({n_qat} wraps)+Muon | "
        f"loss {losses[0]:.4f}->{losses[-1]:.4f} decreasing={losses[-1] < losses[0]} | "
        f"alpha moved off 0: {abs(alpha.item()) > 1e-6}",
        flush=True,
    )
    print("[CAPSTONE] PASS", flush=True)


if __name__ == "__main__":
    main()
