"""Per-rank expert load for both dispatchers on the same tokens.

The balance claim: under routing that sends every token to experts homed on one
rank, MoonEP's prefetch slots let other ranks compute copies, while the standard
all-to-all path leaves the owning rank with all of it.
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

E, K, S, D, F = 32, 4, 256, 512, 384
BETA, LINEAR_BETA = 4.0, 25.0


def reference(x_TD, weights_TK, ids_TK, w1, w2, w3):
    act = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA).build()
    out = torch.zeros_like(x_TD)
    for k in range(K):
        e = ids_TK[:, k]
        gate = torch.einsum("td,tfd->tf", x_TD, w1[e])
        up = torch.einsum("td,tfd->tf", x_TD, w3[e])
        out = out + weights_TK[:, k : k + 1] * torch.einsum(
            "tf,tdf->td", act(gate, up), w2[e]
        )
    return out


def rel(a, b):
    n = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)
    return float(n.median()), float(n.max())


def main():
    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))

    torch.manual_seed(1)
    params = {
        "w1": torch.randn(E, F, D) * 0.05,
        "w2": torch.randn(E, D, F) * 0.05,
        "w3": torch.randn(E, F, D) * 0.05,
    }
    lo, hi = rank * (E // size), (rank + 1) * (E // size)
    act_cfg = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA)

    routing = os.environ.get("ROUTING", "uniform")
    torch.manual_seed(100 + rank)
    x = (torch.randn(S, D, device=dev) * 0.5).to(torch.bfloat16)
    if routing == "hot":
        # Every token to experts 0..K-1, all homed on rank 0.
        ids = torch.arange(K, device=dev, dtype=torch.int64).repeat(S, 1)
        weights = torch.softmax(torch.randn(S, K, device=dev), dim=-1)
    else:
        weights, ids = torch.rand(S, E, device=dev).topk(K, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
    counts = torch.zeros(E, dtype=torch.long, device=dev).scatter_add_(
        0, ids.reshape(-1), torch.ones(S * K, dtype=torch.long, device=dev)
    )

    # --- MoonEP path -------------------------------------------------------
    mp_experts = MoonEPGroupedExperts(
        MoonEPGroupedExperts.Config(
            dim=D, hidden_dim=F, num_experts=E, activation_fn=act_cfg
        )
    ).to(dev)
    for attr, key in (("w1_EFD", "w1"), ("w2_EDF", "w2"), ("w3_EFD", "w3")):
        setattr(mp_experts, attr, nn.Parameter(params[key][lo:hi].to(dev, torch.bfloat16)))
    mp_disp = MoonEPTokenDispatcher(
        MoonEPTokenDispatcher.Config(
            num_experts=E, top_k=K, hidden_dim=D,
            num_max_tokens_per_rank=S, expert_hidden_dim=F,
        )
    )
    mp_disp.wire_meshes(ep_mesh=mesh)
    mp_experts.attach(mp_disp, MoonEPTableBackendNVLink(mesh, mp_disp), mesh)
    routed, rows, meta = mp_disp.dispatch(x, weights, ids, counts)
    out_moonep = mp_disp.combine(mp_experts(routed, rows), meta, x).float()

    # --- standard all-to-all path -----------------------------------------
    std_experts = GroupedExperts(
        GroupedExperts.Config(
            dim=D, hidden_dim=F, num_experts=E // size, activation_fn=act_cfg
        )
    ).to(dev)
    for attr, key in (("w1_EFD", "w1"), ("w2_EDF", "w2"), ("w3_EFD", "w3")):
        setattr(std_experts, attr, nn.Parameter(params[key][lo:hi].to(dev, torch.bfloat16)))
    std_disp = AllToAllTokenDispatcher(
        AllToAllTokenDispatcher.Config(num_experts=E, top_k=K)
    )
    std_disp.wire_meshes(ep_mesh=mesh)
    with set_current_spmd_mesh(mesh):
        routed_s, rows_s, meta_s = std_disp.dispatch(x, weights, ids, counts)
        out_std = std_disp.combine(std_experts(routed_s, rows_s), meta_s, x).float()

    # --- what each rank actually computed ---------------------------------
    lo, hi = rank * (E // size), (rank + 1) * (E // size)
    moonep_rows = int(rows[lo:hi].sum()) + int(rows[E:].sum())
    std_rows = int(rows_s.sum())
    slot_rows = int(rows[E:].sum())

    gathered = torch.zeros(3, size, dtype=torch.int64, device=dev)
    gathered[0, rank] = std_rows
    gathered[1, rank] = moonep_rows
    gathered[2, rank] = slot_rows
    dist.all_reduce(gathered)

    torch.cuda.synchronize()
    if rank == 0:
        std_l = gathered[0].tolist()
        mp_l = gathered[1].tolist()
        sl_l = gathered[2].tolist()
        def imb(v):
            return max(v) / (sum(v) / len(v)) if sum(v) else float("nan")
        print(f"routing={routing}  {size} ranks, {S} tokens/rank, top_k {K}, {E} experts, B={mp_disp.num_prefetch_slots}")
        print(f"  {'rows computed per rank':<26} {'max/mean':>9}   per rank")
        print(f"  {'standard all-to-all':<26} {imb(std_l):>9.2f}   {std_l}")
        print(f"  {'moonep (own + slots)':<26} {imb(mp_l):>9.2f}   {mp_l}")
        print(f"  {'   of which in slots':<26} {'':>9}   {sl_l}")
        print(f"  total rows: standard {sum(std_l)}, moonep {sum(mp_l)}")
    buf = getattr(mp_disp, "_buffer", None)
    if buf is not None and hasattr(buf, "destroy"):
        buf.destroy()
    dist.barrier(); dist.destroy_process_group()


main()
