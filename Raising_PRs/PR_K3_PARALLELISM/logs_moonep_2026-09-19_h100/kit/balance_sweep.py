"""Per-rank expert load for both dispatchers across a sweep of routing imbalance.

Shapes are set from the environment so the same script runs at debug size and at
a size where the static layout's 128-token padding is a fraction of a row.
Routing is drawn from a Zipf distribution over experts, the same hot experts on
every rank, so the imbalance is global rather than per rank.
"""
import os

import torch, torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.kimi_k3.moon_ep_dispatcher import MoonEPTokenDispatcher
from torchtitan.models.kimi_k3.moon_ep_experts import (
    MoonEPGroupedExperts,
    MoonEPTableBackendNVLink,
)

E = int(os.environ.get("NUM_EXPERTS", 256))
K = int(os.environ.get("TOP_K", 8))
S = int(os.environ.get("TOKENS_PER_RANK", 4096))
D = int(os.environ.get("DIM", 1024))
F = int(os.environ.get("EXPERT_HIDDEN", 1024))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0,0.5,1.0,1.5,2.0").split(",")]
BETA, LINEAR_BETA = 4.0, 25.0


def routing(alpha, dev, size):
    """Zipf over experts with a fixed hot order shared by every rank."""
    g = torch.Generator(device="cpu").manual_seed(7)
    order = torch.randperm(E, generator=g)
    p = torch.zeros(E)
    p[order] = (torch.arange(1, E + 1, dtype=torch.float) ** (-alpha))
    p = (p / p.sum()).to(dev)
    ids = torch.multinomial(p.expand(S, E), K, replacement=False)
    w = torch.rand(S, K, device=dev)
    return w / w.sum(-1, keepdim=True), ids


def imb(v):
    return max(v) / (sum(v) / len(v)) if sum(v) else float("nan")


def main():
    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
    act_cfg = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA)
    lo, hi = rank * (E // size), (rank + 1) * (E // size)

    torch.manual_seed(1)
    w1 = (torch.randn(E // size, F, D, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E // size, D, F, device=dev) * 0.05).to(torch.bfloat16)
    w3 = (torch.randn(E // size, F, D, device=dev) * 0.05).to(torch.bfloat16)

    mp_experts = MoonEPGroupedExperts(
        MoonEPGroupedExperts.Config(dim=D, hidden_dim=F, num_experts=E, activation_fn=act_cfg)
    ).to(dev)
    mp_experts.w1_EFD, mp_experts.w2_EDF, mp_experts.w3_EFD = (
        nn.Parameter(w1), nn.Parameter(w2), nn.Parameter(w3)
    )
    mp_disp = MoonEPTokenDispatcher(
        MoonEPTokenDispatcher.Config(
            num_experts=E, top_k=K, hidden_dim=D, num_max_tokens_per_rank=S, expert_hidden_dim=F
        )
    )
    mp_disp.wire_meshes(ep_mesh=mesh)
    mp_experts.attach(mp_disp, MoonEPTableBackendNVLink(mesh, mp_disp), mesh)

    std_disp = AllToAllTokenDispatcher(AllToAllTokenDispatcher.Config(num_experts=E, top_k=K))
    std_disp.wire_meshes(ep_mesh=mesh)

    if rank == 0:
        print(f"E={E} top_k={K} tokens/rank={S} dim={D} expert_hidden={F} "
              f"ranks={size} own={E // size} B={mp_disp.num_prefetch_slots}")
        print(f"  mean tokens per expert = {S * size * K / E:.0f} against a 128 padding unit")
        print(f"{'alpha':>6} {'route max/mean':>15} {'std max/mean':>13} "
              f"{'moonep max/mean':>16} {'std rows':>10} {'moonep rows':>12}")

    for alpha in ALPHAS:
        weights, ids = routing(alpha, dev, size)
        counts = torch.zeros(E, dtype=torch.long, device=dev).scatter_add_(
            0, ids.reshape(-1), torch.ones(S * K, dtype=torch.long, device=dev)
        )
        x = (torch.randn(S, D, device=dev) * 0.5).to(torch.bfloat16)
        _, rows_m, _ = mp_disp.dispatch(x, weights, ids, counts)
        with set_current_spmd_mesh(mesh):
            _, rows_s, _ = std_disp.dispatch(x, weights, ids, counts)

        g = torch.zeros(3, size, dtype=torch.int64, device=dev)
        g[0, rank] = int(rows_s.sum())
        g[1, rank] = int(rows_m[lo:hi].sum()) + int(rows_m[E:].sum())
        global_counts = counts.clone()
        dist.all_reduce(global_counts)
        dist.all_reduce(g)
        if rank == 0:
            s_l, m_l = g[0].tolist(), g[1].tolist()
            route = float(global_counts.max()) / float(global_counts.float().mean())
            print(f"{alpha:>6.1f} {route:>15.2f} {imb(s_l):>13.2f} {imb(m_l):>16.2f} "
                  f"{sum(s_l):>10d} {sum(m_l):>12d}")

    buf = getattr(mp_disp, "_buffer", None)
    if buf is not None and hasattr(buf, "destroy"):
        buf.destroy()
    dist.barrier(); dist.destroy_process_group()


main()
