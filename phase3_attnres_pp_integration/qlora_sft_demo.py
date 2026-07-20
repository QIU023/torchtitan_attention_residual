"""Standalone QLoRA SFT loop (post-load quantize, trainer order):
build 194m graft flavor -> init (stand-in for checkpoint load) -> wrap
LoRA -> quantize_lora_bases (NF4 post-load hook) -> fully_shard(8) ->
train N steps. Reports loss trend + peak memory, comparing bf16-base
LoRA vs NF4-base QLoRA. This is the meta-first trainer order: bases are
packed AFTER weights materialize, not at build over init noise."""
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def run(quantize: bool, steps: int = 15):
    from torchtitan.experiments.kimi_k3 import config_registry
    from torchtitan.experiments.kimi_k3.lora import (
        apply_lora,
        quantize_lora_bases,
    )
    from torchtitan.experiments.kimi_k3.model import KimiLinearSpec

    torch.manual_seed(0)
    kc = config_registry.build_kimi_linear_config("194m", num_experts=32)
    # Build the plain backbone and init (stand-in for checkpoint load),
    # THEN wrap LoRA and NF4-pack the loaded bases -- the meta-first
    # trainer order. lora_quantize_base is deliberately NOT set on the
    # spec (that packs at build over init noise); the post-load hook
    # packs real weights. NF4 lands on the dim-aligned target LINEARS
    # (896 % 64 == 0 at 194m). GroupedExperts NF4 is skipped here: 194m
    # expert dims are not NF4-block-aligned (numel/8 double-quant
    # constraint) -- a real fragility of the experts hack; 48B dims align.
    spec = KimiLinearSpec(
        kimi_config=kc, num_blocks=4, attn_res_gated=True,
    )
    with torch.device("cuda"):
        model = spec.build()
        model.init_weights()
    model = model.to(torch.bfloat16)
    apply_lora(model, rank=8, alpha=16)
    if quantize:
        quantize_lora_bases(model, experts=False)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for layer in model.layers.values():
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-4)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(steps):
        tok = torch.randint(0, kc.vocab_size, (1, 256), device="cuda")
        out = model(tok)
        loss = out.float().pow(2).mean()  # dummy objective; exercises bwd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    return losses, peak, len(trainable)


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    for label, q in (("bf16-base LoRA", False), ("NF4-base QLoRA", q := True)):
        losses, peak, nt = run(q)
        if rank == 0:
            print(
                f"[QLORA-SFT] {label}: steps={len(losses)} "
                f"loss {losses[0]:.4f}->{losses[-1]:.4f} "
                f"peak={peak:.2f} GiB trainable_tensors={nt}",
                flush=True,
            )
        dist.barrier()
    if rank == 0:
        print("[QLORA-SFT] PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
