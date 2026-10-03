"""Run MoonEPRoutedExperts like tests/unit_tests/gpu/test_moonep.py and dump outputs and gradients.

torchrun --nproc_per_node=N bitwise_dump.py --out DIR [--layers L] [--mbs M] [--routing hot|hot2|uniform]
    [--ac none|selective|full|region] [--shared]

Each rank writes DIR/rank{r}.pt with the outputs, input gradients and expert weight gradients, so two trees
(old and new head) can be compared with torch.equal. --shared adds a dense shared expert to every layer so
the shared-experts stream path is exercised as well.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.config.transform import convert_config_type
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC
from torchtitan.distributed.moonep import ops
from torchtitan.distributed.moonep.experts import MoonEPRoutedExperts
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher
from torchtitan.protocols.module import Module, ModuleDict

E, K, S, D, F = 32, 4, 256, 512, 384
BETA, LINEAR_BETA = 4.0, 25.0


class _Block(Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, *args):
        return self.experts(*args)


class _Model(Module):
    def __init__(self, experts):
        super().__init__()
        self.layers = ModuleDict({str(i): _Block(e) for i, e in enumerate(experts)})


def _routing(kind, home, size, device):
    if kind == "uniform":
        weights_TK, ids_TK = torch.rand(S, E, device=device).topk(K, dim=-1)
        return weights_TK / weights_TK.sum(-1, keepdim=True), ids_TK
    homes = [home] if kind == "hot" else [home, (home + 1) % size]
    ids_K = torch.cat([torch.arange(K // len(homes), device=device) + h * (E // size) for h in homes])
    return torch.softmax(torch.randn(S, K, device=device), dim=-1), ids_K.repeat(S, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--routing", default="hot")
    p.add_argument("--ac", default="none")
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
    set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
    lo, hi = rank * (E // size), (rank + 1) * (E // size)

    layers = []
    for layer in range(args.layers):
        torch.manual_seed(1 + layer)
        w13_E2FD = torch.randn(E, 2, F, D) * 0.05
        w2_EDF = torch.randn(E, D, F) * 0.05
        config = convert_config_type(
            make_routed_experts_config(dim=D, hidden_dim=F, num_experts=E, top_k=K, param_init={}),
            MoonEPRoutedExperts,
        )
        config.token_dispatcher = MoonEPTokenDispatcher.Config(
            num_experts=E, top_k=K, hidden_dim=D, num_max_tokens_per_rank=S
        )
        config.activation_fn = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA)
        experts = config.build().to(device)
        experts.w13.weight = nn.Parameter(w13_E2FD[lo:hi].to(device, torch.bfloat16))
        experts.w2.weight = nn.Parameter(w2_EDF[lo:hi].to(device, torch.bfloat16))
        layers.append(experts)
    model = _Model(layers)
    ac = {
        "none": None,
        "selective": SelectiveAC.Config(),
        "full": FullAC.Config(),
        "region": RegionAC.Config(save_regions=[]),
    }[args.ac]
    if ac is not None:
        ac.build().apply(model)

    microbatches = []
    for mb in range(args.mbs):
        torch.manual_seed(100 + rank + size * mb)
        x_TD = (torch.randn(S, D, device=device) * 0.5).to(torch.bfloat16)
        routes = [_routing(args.routing, (mb + layer) % size, size, device) for layer in range(args.layers)]
        for weights_TK, _ in routes:
            weights_TK.requires_grad_(True)
        microbatches.append((x_TD, routes))

    x_ins, outs = [], []
    first_plan = ops._next_plan_id
    with set_current_spmd_mesh(mesh):
        for experts in layers:
            experts.token_dispatcher.init_buffer()
        for x_TD, routes in microbatches:
            h_TD = x_TD.clone().requires_grad_(True)
            x_ins.append(h_TD)
            for layer, (weights_TK, ids_TK) in enumerate(routes):
                counts_E = torch.bincount(ids_TK.flatten(), minlength=E)
                h_TD = model.layers[str(layer)](h_TD, weights_TK, ids_TK, counts_E)
            outs.append(h_TD)
        for out_TD in outs:
            out_TD.float().sum().backward()
    torch.cuda.synchronize()
    assert ops._next_plan_id - first_plan == args.layers * args.mbs, "a dispatch was replayed"
    assert ops._plans == {}, "a plan outlived its combine"
    dump = {
        "out": [o.detach().cpu() for o in outs],
        "grad_x": [x.grad.cpu() for x in x_ins],
        "grad_w13": [e.w13.weight.grad.cpu() for e in layers],
        "grad_w2": [e.w2.weight.grad.cpu() for e in layers],
        "grad_weights": [w.grad.cpu() for _, routes in microbatches for w, _ in routes],
        "peak_alloc": torch.cuda.max_memory_allocated(),
    }
    gathered_mbs = []
    for x_TD, routes in microbatches:
        tensors = [x_TD.detach(), *[t.detach() for route in routes for t in route]]
        gathered = []
        for t in tensors:
            parts = [torch.empty_like(t) for _ in range(size)]
            dist.all_gather(parts, t)
            gathered.append(torch.cat(parts))
        gathered_mbs.append(gathered)
    refs = []
    for layer in range(args.layers):
        torch.manual_seed(1 + layer)
        w13_E2FD = torch.randn(E, 2, F, D) * 0.05
        w2_EDF = torch.randn(E, D, F) * 0.05
        refs.append(tuple(w.to(device, torch.bfloat16).float().requires_grad_(True) for w in (w13_E2FD, w2_EDF)))
    activation = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA).build()
    mine = slice(rank * S, (rank + 1) * S)
    ref_out, ref_gx, ref_gw = [], [], []
    for gathered in gathered_mbs:
        x_all = gathered[0].float().requires_grad_(True)
        w_all = [gathered[1 + 2 * l].float().requires_grad_(True) for l in range(args.layers)]
        h = x_all
        for l, (w13, w2) in enumerate(refs):
            ids = gathered[2 + 2 * l]
            out = torch.zeros_like(h)
            for e in ids.unique().tolist():
                t, k = (ids == e).nonzero(as_tuple=True)
                hid = activation(h[t] @ w13[e, 0].T, h[t] @ w13[e, 1].T)
                out = out.index_add(0, t, w_all[l][t, k, None] * (hid @ w2[e].T))
            h = out
        h.sum().backward()
        ref_out.append(h[mine].detach().cpu())
        ref_gx.append(x_all.grad[mine].cpu())
        ref_gw.extend(w.grad[mine].cpu() for w in w_all)
    dump["ref"] = {
        "out": ref_out,
        "grad_x": ref_gx,
        "grad_w13": [w13.grad[lo:hi].cpu() for w13, _ in refs],
        "grad_w2": [w2.grad[lo:hi].cpu() for _, w2 in refs],
        "grad_weights": ref_gw,
    }
    os.makedirs(args.out, exist_ok=True)
    torch.save(dump, os.path.join(args.out, f"rank{rank}.pt"))
    layers[0].token_dispatcher.buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
