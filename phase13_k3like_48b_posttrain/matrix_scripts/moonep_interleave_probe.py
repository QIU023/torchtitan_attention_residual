"""Does a later micro-batch's forward corrupt an earlier one's backward?

The expert tables are per-module state: a forward refreshes this rank's rows and
prefetches the slots for its own plan, and the backward recomputes from whatever
the tables hold at backward time. A pipeline schedule runs the forward of
micro-batch i+1 before the backward of micro-batch i on every stage but the last.

Sequential runs fwd(mb0) bwd(mb0) fwd(mb1) bwd(mb1); interleaved runs
fwd(mb0) fwd(mb1) bwd(mb0) bwd(mb1). The two routings differ so the plans pick
different slots. Same weights, same inputs, so any difference in mb0's
gradients is the corruption.
"""
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.distributed.moonep.moonep import MoonEPTableBackendNVLink
from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.moe import MoonEPGroupedExperts
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher

E, K, S, D, F = 32, 4, 256, 512, 384
BETA, LINEAR_BETA = 1.0, 1.0


def build(dev, mesh, rank, size):
    torch.manual_seed(1)
    params = {
        "w1": torch.randn(E, F, D) * 0.05,
        "w2": torch.randn(E, D, F) * 0.05,
        "w3": torch.randn(E, F, D) * 0.05,
    }
    experts = MoonEPGroupedExperts(
        MoonEPGroupedExperts.Config(
            dim=D, hidden_dim=F, num_experts=E,
            activation_fn=SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA),
        )
    ).to(dev)
    lo, hi = rank * (E // size), (rank + 1) * (E // size)
    experts.w1_EFD = nn.Parameter(params["w1"][lo:hi].to(dev, torch.bfloat16))
    experts.w2_EDF = nn.Parameter(params["w2"][lo:hi].to(dev, torch.bfloat16))
    experts.w3_EFD = nn.Parameter(params["w3"][lo:hi].to(dev, torch.bfloat16))
    dispatcher = MoonEPTokenDispatcher(
        MoonEPTokenDispatcher.Config(
            num_experts=E, top_k=K, hidden_dim=D,
            num_max_tokens_per_rank=S, expert_hidden_dim=F,
        )
    )
    dispatcher.wire_meshes(ep_mesh=mesh)
    experts.attach(dispatcher, MoonEPTableBackendNVLink(mesh, dispatcher), mesh)
    return experts, dispatcher


def batch(dev, rank, routing):
    torch.manual_seed(100 + rank + (0 if routing == "hot" else 7))
    x = (torch.randn(S, D, device=dev) * 0.5).to(torch.bfloat16)
    if routing == "hot":
        ids = torch.arange(K, device=dev, dtype=torch.int64).repeat(S, 1)
        weights = torch.softmax(torch.randn(S, K, device=dev), dim=-1)
    else:
        weights, ids = torch.rand(S, E, device=dev).topk(K, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
    counts = torch.zeros(E, dtype=torch.long, device=dev).scatter_add_(
        0, ids.reshape(-1), torch.ones(S * K, dtype=torch.long, device=dev)
    )
    return x, weights, ids, counts


def forward(experts, dispatcher, b, dev):
    x, weights, ids, counts = b
    x_in = x.clone().requires_grad_(True)
    routed, rows, metadata = dispatcher.dispatch(x_in, weights, ids, counts)
    out = dispatcher.combine(experts(routed, rows), metadata, x_in)
    torch.manual_seed(5)
    g = torch.randn_like(out)
    return x_in, out, g


def grads(experts, x_in):
    return (
        x_in.grad.detach().float().clone(),
        experts.w1_EFD.grad.detach().float().clone(),
        experts.w2_EDF.grad.detach().float().clone(),
    )


def main():
    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
    b0 = batch(dev, rank, "hot")
    b1 = batch(dev, rank, "uniform")

    experts, dispatcher = build(dev, mesh, rank, size)
    x0, out0, g0 = forward(experts, dispatcher, b0, dev)
    out0.backward(g0)
    seq = grads(experts, x0)
    plan_slots_hot = dispatcher.current_plan()[0].experts_to_copy.clone()

    experts2, dispatcher2 = build(dev, mesh, rank, size)
    y0, o0, gg0 = forward(experts2, dispatcher2, b0, dev)
    y1, o1, gg1 = forward(experts2, dispatcher2, b1, dev)   # the next micro-batch's forward
    plan_slots_uniform = dispatcher2.current_plan()[0].experts_to_copy.clone()
    o0.backward(gg0)                                         # then the earlier backward
    inter = grads(experts2, y0)

    if rank == 0:
        same_plan = bool(torch.equal(plan_slots_hot, plan_slots_uniform))
        print(f"the two plans pick the same slots: {same_plan}")
        print(f"hot slots     {plan_slots_hot.tolist()}")
        print(f"uniform slots {plan_slots_uniform.tolist()}")
        for name, a, b in zip(("dgrad(x)", "wgrad(w1)", "wgrad(w2)"), seq, inter):
            d = (a - b).abs().max().item()
            rel = d / max(a.abs().max().item(), 1e-9)
            print(f"{name:10s} max abs diff {d:.6e}  rel {rel:.3e}  {'SAME' if d == 0 else 'DIFFERENT'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
