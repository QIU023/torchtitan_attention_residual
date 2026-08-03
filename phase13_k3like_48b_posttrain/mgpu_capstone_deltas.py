"""Task 23: the K3 deltas composed under FSDP2 on N GPUs (the capstone
smoke was single-GPU). Gated MLA + alpha-graft AttnRes + MXFP4/MXFP8 QAT
fake-quant + Per-Head Muon, all trained together under fully_shard.

Also a real question: Muon does Newton-Schulz orthogonalization of the
momentum, which needs the full 2-D matrix -- under FSDP the params are
sharded DTensors. This checks whether Muon composes with FSDP2 sharding.
"""
import dataclasses
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def main():
    from torchtitan.experiments.kimi_k3 import config_registry
    from torchtitan.experiments.kimi_k3.model import KimiK3Spec
    from torchtitan.experiments.kimi_k3.mxfp4_qat import apply_mxfp4_qat
    from torchtitan.experiments.kimi_k3.muon import Muon

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    torch.manual_seed(0)

    kc = config_registry.build_kimi_linear_config("194m", num_experts=32)
    kc = dataclasses.replace(kc, mla_gated=True)  # Gated MLA
    spec = KimiK3Spec(kimi_config=kc, num_blocks=4, attn_res_gated=True)
    with torch.device("cuda"):
        model = spec.build()
        model.init_weights()
    n_qat = apply_mxfp4_qat(model, quantize_act=True)  # MXFP4/MXFP8 QAT
    model = model.to(torch.bfloat16)
    heads = kc.num_attention_heads
    for name, p in model.named_parameters():
        if name.endswith("q_proj.base.weight") or name.endswith("o_proj.base.weight"):
            p._muon_heads = heads

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for layer in model.layers.values():
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = Muon(trainable, lr=1e-3, adamw_lr=2e-4)  # Per-Head Muon
    losses = []
    for _ in range(5):
        tok = torch.randint(0, kc.vocab_size, (1, 256), device="cuda")
        out = model(tok)
        loss = out.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    finite = all(torch.isfinite(torch.tensor(x)) for x in losses)
    if rank == 0:
        print(
            f"[MGPU-CAPSTONE] GatedMLA+AlphaGraft+MXFP4QAT({n_qat})+Muon "
            f"under FSDP2/{dist.get_world_size()}gpu: "
            f"loss {losses[0]:.4f}->{losses[-1]:.4f} finite={finite}",
            flush=True,
        )
        print("[MGPU-CAPSTONE] PASS" if finite else "[MGPU-CAPSTONE] FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
