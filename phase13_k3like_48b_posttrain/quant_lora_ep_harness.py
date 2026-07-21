"""Direct verification: NF4 and MXFP4 base LoRA under FSDP+EP (4 GPU).

torchtitan.train can't run quantized LoRA (meta-first shard-then-init vs
the required quantize-then-shard); this harness uses the validated
build-on-device order and the real EP parallelize path. MXFP4/NF4 quantize
the KimiLoRALinear bases (attn/gate/shared-expert linears, FSDP-sharded);
EP shards the separate routed GroupedExperts (bf16) -- orthogonal
subsystems, verified composing here.
"""
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def main(mode):
    from torchtitan.distributed.parallel_dims import ParallelDims
    from torchtitan.experiments.kimi_k3 import config_registry
    from torchtitan.experiments.kimi_k3.lora import (
        KimiLoRALinear,
        apply_lora,
        quantize_lora_bases,
    )
    from torchtitan.experiments.kimi_k3.model import KimiLinearSpec
    from torchtitan.experiments.kimi_k3.parallelize import apply_ep_kimi_linear

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    r = dist.get_rank()
    ws = dist.get_world_size()
    # ep carved from dp_shard: dp_shard=ws, ep=2.
    pd = ParallelDims(
        dp_replicate=1, dp_shard=ws, cp=1, tp=1, pp=1, ep=2, world_size=ws
    )
    pd.build_mesh()

    torch.manual_seed(0)
    cfg = config_registry.kimi_linear_debugmodel_gated_lora()
    m = cfg.model_spec.model
    # EP needs num_experts % ep == 0 (debug has 8); enable ep flag for build.
    m.kimi_config.moe_enable_ep = True
    with torch.device("cuda"):
        model = m.build()
        model.init_weights()
    model = model.to(torch.bfloat16)
    n_q = quantize_lora_bases(model, mode=mode, experts=False)  # BEFORE shard

    apply_ep_kimi_linear(model, pd)  # EP on routed experts
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    fsdp_mesh = pd.get_mesh("fsdp")
    for layer in model.layers.values():
        fully_shard(layer, mesh=fsdp_mesh, mp_policy=mp)
    fully_shard(model, mesh=fsdp_mesh, mp_policy=mp)

    q_bases = sum(
        1 for mod in model.modules()
        if isinstance(mod, KimiLoRALinear) and mod._quantize_base == mode
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    losses = []
    for _ in range(3):
        tok = torch.randint(0, m.kimi_config.vocab_size, (1, 96), device="cuda")
        loss = model(tok).float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    if r == 0:
        fin = all(torch.isfinite(torch.tensor(x)) for x in losses)
        print(
            f"[QUANT-LORA-EP {mode}] q_bases={q_bases} ep={pd.ep} "
            f"loss {losses[0]:.4f}->{losses[-1]:.4f} finite={fin}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nf4")
