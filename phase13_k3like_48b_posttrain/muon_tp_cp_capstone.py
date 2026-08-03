"""B7 capstone: Per-Head Muon under FSDP x TP x CP (2026-07-24).

EIGHTCARD verified Muon's Newton-Schulz on FSDP2-sharded DTensors; the
full-param K3 recipe wants it under FSDP x TP (x CP) where params are
2-D-mesh DTensors (fsdp x tp) and grads carry CP-reduced contributions.
This runs the real parallelize_kimi_k3 wiring (dp2 x tp2 x cp2 = 8
ranks) on the gated debugmodel, trains 5 steps with Muon on the
q/o_proj matrices (per-head) + AdamW fallback for the rest, and gates
on finite descending loss.

Launch: torchrun --nproc_per_node=8 muon_tp_cp_capstone.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main() -> None:
    from torchtitan.config import (
        CompileConfig,
        ParallelismConfig,
        TrainingConfig,
    )
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.kimi_k3 import config_registry
    from torchtitan.experiments.kimi_k3.model import KimiK3Spec
    from torchtitan.experiments.kimi_k3.muon import Muon
    from torchtitan.experiments.kimi_k3.parallelize import (
        parallelize_kimi_k3,
    )

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    torch.manual_seed(0)

    import dataclasses

    kc = config_registry.build_kimi_linear_config(
        "debugmodel8h", num_experts=8, vocab_size=2016,
    )
    # trainer's update_from_config equivalent: MoE must know about TP at
    # build time (module-internal parallelization).
    kc = dataclasses.replace(kc, moe_enable_tp=True)
    spec = KimiK3Spec(kimi_config=kc, num_blocks=4)
    with torch.device("cuda"):
        model = spec.build()
        model.init_weights()

    pd = ParallelDims(
        dp_shard=2, dp_replicate=1, cp=2, tp=2, pp=1, ep=1, world_size=8,
    )
    pd.build_mesh()
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        context_parallel_degree=2,
        context_parallel_load_balancer=None,
    )
    model = parallelize_kimi_k3(
        model,
        parallel_dims=pd,
        training=TrainingConfig(),
        parallelism=parallelism,
        compile_config=CompileConfig(enable=False),
        ac_config=None,
        dump_folder="/tmp/muon_tp_cp_capstone",
    )

    heads = kc.num_attention_heads
    for name, p in model.named_parameters():
        if name.endswith("q_proj.weight") or name.endswith("o_proj.weight"):
            p._muon_heads = heads

    opt = Muon([p for p in model.parameters() if p.requires_grad], lr=1e-3, adamw_lr=2e-4)

    cp_rank = dist.get_rank(pd.get_mesh("cp").get_group())
    T, t_loc = 256, 256 // 2
    losses = []
    for step in range(5):
        tok_full = torch.randint(0, kc.vocab_size, (1, T), device="cuda")
        # trainer contract: seq-sharded contiguous slice per cp rank
        tok = tok_full[:, cp_rank * t_loc : (cp_rank + 1) * t_loc]
        out = model(tok)
        loss = F.cross_entropy(
            out.float().view(-1, out.shape[-1]),
            tok.reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    finite = all(torch.isfinite(torch.tensor(x)) for x in losses)
    if rank == 0:
        print(
            f"[MUON-TP-CP] FSDP2xTP2xCP2 Muon: loss "
            f"{losses[0]:.4f}->{losses[-1]:.4f} finite={finite}",
            flush=True,
        )
        print("[MUON-TP-CP] PASS" if finite and losses[-1] < losses[0] else "[MUON-TP-CP] FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
