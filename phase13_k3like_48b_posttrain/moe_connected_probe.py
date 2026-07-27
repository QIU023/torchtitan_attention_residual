"""Do the routed experts actually affect the loss and receive gradients?

An offline single-process forward showed ZERO logit change when every routed
expert weight was zeroed, which would mean the routed path is disconnected.
The suspicion is that the token dispatcher needs an initialized process group,
so this repeats the control under dist with the real apply_fsdp.

Three measurements, in order of what they rule out:
  1. loss(base) vs loss(experts zeroed) -- is the routed path connected at all
  2. loss(base) vs loss(MXFP4 QAT)      -- does the fake-quant reach the math
  3. w1_EFD grad norm                    -- do the experts train
"""

import os

import torch
import torch.distributed as dist

from torchtitan.experiments.kimi_k3.model import KimiLinearModel
from torchtitan.experiments.kimi_k3.model_configs import build_kimi_linear_config
from torchtitan.experiments.kimi_k3.moe import KimiSiTUGroupedExperts
from torchtitan.experiments.kimi_k3.mxfp4_qat import apply_mxfp4_qat
from torchtitan.experiments.kimi_k3.parallelize import apply_fsdp


def build(qat: bool, zero_experts: bool):
    torch.manual_seed(0)
    cfg = build_kimi_linear_config("k3mini", vocab_size=256)
    with torch.device("meta"):
        model = KimiLinearModel(cfg)
    if qat:
        apply_mxfp4_qat(model)
    mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
    )
    apply_fsdp(
        model, mesh, param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32, pp_enabled=False,
    )
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    model.init_weights(buffer_device="cuda")
    if zero_experts:
        with torch.no_grad():
            for _, m in model.named_modules():
                if isinstance(m, KimiSiTUGroupedExperts):
                    for p in m.parameters():
                        p.zero_()
    return model


def step(model):
    torch.manual_seed(1234)
    tokens = torch.randint(0, 256, (1, 256), device="cuda")
    logits = model(tokens).float()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 256), tokens.view(-1)
    )
    loss.backward()
    gnorm = 0.0
    for _, m in model.named_modules():
        if isinstance(m, KimiSiTUGroupedExperts):
            for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
                p = m._parameters.get(name)
                if p is not None and p.grad is not None:
                    g = p.grad
                    g = g.to_local() if hasattr(g, "to_local") else g
                    gnorm += g.float().pow(2).sum().item()
    return loss.item(), gnorm**0.5


def main() -> None:
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()

    results = {}
    for tag, kw in (
        ("base", dict(qat=False, zero_experts=False)),
        ("zeroed", dict(qat=False, zero_experts=True)),
        ("qat", dict(qat=True, zero_experts=False)),
    ):
        results[tag] = step(build(**kw))

    if rank == 0:
        for tag, (loss, gnorm) in results.items():
            print(f"[MOE] {tag:7} loss {loss:.6f}  expert grad-norm {gnorm:.4e}",
                  flush=True)
        base = results["base"][0]
        print(
            f"[MOE] connected (base vs zeroed): "
            f"{abs(base - results['zeroed'][0]):.6e}", flush=True
        )
        print(
            f"[MOE] qat reaches math (base vs qat): "
            f"{abs(base - results['qat'][0]):.6e}", flush=True
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
