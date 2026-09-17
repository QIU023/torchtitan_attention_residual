"""MoonEP through the Kimi K3 dispatcher and expert tables on real GPUs, R ranks under torchrun.

Two routings per run: forced-hot (every token to experts 0..K-1, all home on rank 0, so the other
ranks' prefetch slots must fill) and uniform random. Each is checked against a dense fp32 reference
on the gathered tokens: outputs, the input gradient and the local expert-row gradients.
"""
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.models.kimi_k3.moon_ep_dispatcher import MoonEPTokenDispatcher
from torchtitan.models.kimi_k3.moon_ep_experts import MoonEPGroupedExperts, MoonEPTableBackendNVLink

from torchtitan.models.common.activation import SiTUGLU

E, K, S, D, F = int(os.environ.get("HP_E", 32)), int(os.environ.get("HP_K", 4)), 256, 512, 384
BETA, LBETA = 4.0, 25.0


def reference(x_all, w_all, ids_all, w1, w2, w3):
    act = SiTUGLU.Config(beta=BETA, linear_beta=LBETA).build()
    out = torch.zeros_like(x_all)
    for k in range(K):
        e = ids_all[:, k]
        g = torch.einsum("td,tfd->tf", x_all, w1[e])
        u = torch.einsum("td,tfd->tf", x_all, w3[e])
        h = act(g, u)
        out = out + w_all[:, k : k + 1] * torch.einsum("tf,tdf->td", h, w2[e])
    return out


def run(rank, R, mesh, params, routing, tag):
    torch.manual_seed(100 + rank)
    dev = torch.device("cuda", torch.cuda.current_device())
    experts = MoonEPGroupedExperts(
        MoonEPGroupedExperts.Config(
            dim=D, hidden_dim=F, num_experts=E, activation_fn=SiTUGLU.Config(beta=BETA, linear_beta=LBETA)
        )
    ).to(dev)
    lo, hi = rank * (E // R), (rank + 1) * (E // R)
    experts.w1_EFD = nn.Parameter(params["w1"][lo:hi].to(dev, torch.bfloat16))
    experts.w2_EDF = nn.Parameter(params["w2"][lo:hi].to(dev, torch.bfloat16))
    experts.w3_EFD = nn.Parameter(params["w3"][lo:hi].to(dev, torch.bfloat16))
    dispatcher = MoonEPTokenDispatcher(
        MoonEPTokenDispatcher.Config(num_experts=E, top_k=K, hidden_dim=D, num_max_tokens_per_rank=S)
    )
    dispatcher.wire_meshes(ep_mesh=mesh)
    experts.attach(dispatcher, MoonEPTableBackendNVLink(mesh), mesh)
    x = (torch.randn(S, D, device=dev) * 0.5).to(torch.bfloat16)
    if routing == "hot":
        ids = torch.arange(K, device=dev, dtype=torch.int64).repeat(S, 1)
        weights = torch.softmax(torch.randn(S, K, device=dev), dim=-1)
    else:
        weights, ids = torch.rand(S, E, device=dev).topk(K, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
    counts = torch.zeros(E, dtype=torch.long, device=dev).scatter_add_(0, ids.reshape(-1), torch.ones(S * K, dtype=torch.long, device=dev))
    x_in = x.clone().requires_grad_(True)
    routed, rows, metadata = dispatcher.dispatch(x_in, weights, ids, counts)
    plan, cu = dispatcher.current_plan()
    slots = plan.experts_to_copy[rank].tolist()
    rows_in_slots = int(rows[E:].sum().item())
    expert_out = experts(routed, rows)
    out = dispatcher.combine(expert_out, metadata, x_in)
    out.float().sum().backward()
    torch.cuda.synchronize()
    # dense reference on the gathered tokens
    gx = [torch.empty_like(x) for _ in range(R)]; gw = [torch.empty_like(weights) for _ in range(R)]; gi = [torch.empty_like(ids) for _ in range(R)]
    dist.all_gather(gx, x); dist.all_gather(gw, weights); dist.all_gather(gi, ids)
    x_all = torch.cat(gx).float().requires_grad_(True); w_all = torch.cat(gw).float(); ids_all = torch.cat(gi)
    p = {n: params[n].to(dev, torch.bfloat16).float().requires_grad_(True) for n in params}
    ref = reference(x_all, w_all, ids_all, p["w1"], p["w2"], p["w3"])
    ref.sum().backward()
    my = slice(rank * S, (rank + 1) * S)
    d_out = (out.float() - ref[my]).abs().max().item()
    d_gx = (x_in.grad.float() - x_all.grad[my]).abs().max().item()
    d_w = {n: (getattr(experts, a).grad.float() - p[n].grad[lo:hi]).abs().max().item() for n, a in (("w1", "w1_EFD"), ("w2", "w2_EDF"), ("w3", "w3_EFD"))}
    scale = ref[my].abs().max().item()
    line = (f"HOT_PROBE {tag} rank {rank}/{R} routing={routing} slots={slots} rows_in_slots={rows_in_slots} "
            f"out_maxdiff={d_out:.4f} (ref max {scale:.3f}) gradx_maxdiff={d_gx:.4f} "
            f"w1={d_w['w1']:.4f} w2={d_w['w2']:.4f} w3={d_w['w3']:.4f}")
    print(line, flush=True)
    populated = torch.tensor([any(s >= 0 for s in slots)], device=dev, dtype=torch.int32)
    dist.all_reduce(populated, op=dist.ReduceOp.MAX)
    ok = d_out < 5e-2 * max(1.0, scale) and d_gx < 5e-2 and max(d_w.values()) < 5e-1
    okt = torch.tensor([int(ok)], device=dev, dtype=torch.int32); dist.all_reduce(okt, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"HOT_PROBE_RESULT routing={routing} slot_populated_somewhere={bool(populated.item())} numerics_ok_all_ranks={bool(okt.item())}", flush=True)
    return dispatcher


def main():
    dist.init_process_group("nccl")
    rank, R = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", (R,), mesh_dim_names=("ep",))
    torch.manual_seed(1)
    params = {"w1": torch.randn(E, F, D) * 0.05, "w2": torch.randn(E, D, F) * 0.05, "w3": torch.randn(E, F, D) * 0.05}
    for routing in ("hot", "uniform"):
        d = run(rank, R, mesh, params, routing, f"E{E}K{K}")
        buf = getattr(d, "_buffer", None)
        if buf is not None and hasattr(buf, "destroy"):
            buf.destroy()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
