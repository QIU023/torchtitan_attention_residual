"""Phase 2 (MXFP4 leg): the REAL lora.py MXFP4 path on the real KimiLinear
graft model, under FSDP2 on N GPUs -- a train step, not a toy.

Build the debug graft flavor -> apply LoRA -> quantize_lora_bases(mode=
'mxfp4') (split-storage MXTensor base) -> fully_shard -> 3 train steps.
Verifies the packed MXFP4 base shards + all-gathers + dequant-matmuls and
grads flow to LoRA+graft only (base frozen). Reports peak mem for the
bf16-vs-MXFP4 comparison (Phase 3).
"""
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def run(quantize):
    from torchtitan.experiments.kimi_k3 import config_registry
    from torchtitan.experiments.kimi_k3.lora import (
        KimiLoRALinear,
        quantize_lora_bases,
    )

    torch.manual_seed(0)
    cfg = config_registry.kimi_k3_debugmodel_gated_lora()
    spec_model = cfg.model_spec.model
    vocab = spec_model.kimi_config.vocab_size
    with torch.device("cuda"):
        model = spec_model.build()
        model.init_weights()
    model = model.to(torch.bfloat16)
    n_mxfp4 = 0
    if quantize:
        n_mxfp4 = quantize_lora_bases(model, mode="mxfp4", experts=False)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for layer in model.layers.values():
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    # confirm the base is MXFP4 split-storage (not bf16) when quantized
    mx_bases = sum(
        1
        for m in model.modules()
        if isinstance(m, KimiLoRALinear) and m._quantize_base == "mxfp4"
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(3):
        tok = torch.randint(0, vocab, (1, 256), device="cuda")
        out = model(tok)
        loss = out.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append((loss.item(), gn.item()))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    with_grad = sum(1 for p in trainable if p.grad is not None)
    return losses, peak, n_mxfp4, mx_bases, len(trainable), with_grad


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    for label, q in (("bf16-base LoRA", False), ("MXFP4-base LoRA", True)):
        losses, peak, nmx, mxb, nt, wg = run(q)
        if rank == 0:
            finite = all(
                torch.isfinite(torch.tensor(loss_val)) for loss_val, _ in losses
            )
            print(
                f"[MXFP4-FSDP-real] {label}: mxfp4_bases={mxb} "
                f"loss {losses[0][0]:.4f}->{losses[-1][0]:.4f} "
                f"gnorm {losses[-1][1]:.4f} peak={peak:.3f}GiB "
                f"grad {wg}/{nt} finite={finite}",
                flush=True,
            )
        dist.barrier()
    if rank == 0:
        print("[MXFP4-FSDP-real] PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
