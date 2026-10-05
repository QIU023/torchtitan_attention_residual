"""Time one MoonEP MoE layer per rank: forward plus backward, median over iterations, and the peak memory.

torchrun --nproc_per_node=N microbench.py --out FILE [--mode experts|moe] [--S 4096] [--K 8] [--E 32]
    [--latent 3584] [--hidden 3072] [--shared-hidden 3072] [--layers 2] [--iters 20] [--stream]

--mode experts runs MoonEPRoutedExperts alone on the latent width (the expert op, its prefetch, refill and
reduction); --mode moe runs a whole MoE layer on the latent width (router, MoonEP routed experts, a shared
expert of 2 x --shared-hidden standing in for K3's two), so --stream can put the shared experts on a side
stream; it routes with its own router, so --routing applies to --mode experts only. All parameters are bf16. Layers are stacked so that the shared pools are refilled as in a model.
Rank 0 writes FILE (json).
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.config.transform import convert_config_type
from torchtitan.distributed.moonep.experts import MoonEPRoutedExperts
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.activation import Sigmoid, SiTUGLU
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher


def _routed_config(args, size):
    config = convert_config_type(
        make_routed_experts_config(
            dim=args.latent, hidden_dim=args.hidden, num_experts=args.E, top_k=args.K, param_init={}
        ),
        MoonEPRoutedExperts,
    )
    config.token_dispatcher = MoonEPTokenDispatcher.Config(
        num_experts=args.E, top_k=args.K, hidden_dim=args.latent, num_max_tokens_per_rank=args.S
    )
    config.activation_fn = SiTUGLU.Config(beta=4.0, linear_beta=25.0)
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="experts")
    p.add_argument("--S", type=int, default=4096)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--E", type=int, default=32)
    p.add_argument("--latent", type=int, default=3584)
    p.add_argument("--hidden", type=int, default=3072)
    p.add_argument("--shared-hidden", type=int, default=3072)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--routing", default="uniform")
    p.add_argument("--profile", default="", help="write a per-kernel table of 3 more iterations here (rank 0)")
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
    set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
    lo, hi = rank * (args.E // size), (rank + 1) * (args.E // size)

    layers = []
    for layer in range(args.layers):
        torch.manual_seed(1 + layer)
        if args.mode == "experts":
            module = _routed_config(args, size).build().to(device=device, dtype=torch.bfloat16)
        else:
            config = make_moe_config(
                num_experts=args.E,
                router=make_router_config(
                    dim=args.latent, num_experts=args.E, gate_param_init={}, score_func=Sigmoid.Config(), top_k=args.K
                ),
                routed_experts=_routed_config(args, size),
                shared_experts=make_shared_expert_ffn_config(
                    dim=args.latent, hidden_dim=2 * args.shared_hidden, w1_param_init={}, w2w3_param_init={}
                ),
                load_balance_coeff=None,
            )
            if args.stream:
                config.shared_experts_stream = True
            module = config.build().to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            for param in module.parameters():
                param.copy_(torch.randn_like(param, dtype=torch.float32).mul_(0.02).to(param.dtype))
        routed = module if args.mode == "experts" else module.routed_experts
        for linear in (routed.w13, routed.w2):
            local = linear.weight[lo:hi].detach().clone()
            linear.weight = nn.Parameter(local)
        layers.append(module)
    experts = [m if args.mode == "experts" else m.routed_experts for m in layers]

    torch.manual_seed(100 + rank)
    x_TD = (torch.randn(args.S, args.latent, device=device) * 0.5).to(torch.bfloat16)
    routes = []
    for layer in range(args.layers):
        scores_TE = torch.rand(args.S, args.E, device=device)
        if args.routing == "skew":
            # Every token picks the first K/2 experts of one rank, a different rank per layer.
            first = (layer % size) * (args.E // size)
            scores_TE[:, first : first + args.K // 2] += 1.0
        weights_TK, ids_TK = scores_TE.topk(args.K, dim=-1)
        routes.append((weights_TK / weights_TK.sum(-1, keepdim=True), ids_TK))

    def step():
        h = x_TD.clone().requires_grad_(True)
        for module, (weights_TK, ids_TK) in zip(layers, routes):
            if args.mode == "experts":
                counts_E = torch.bincount(ids_TK.flatten(), minlength=args.E)
                h = module(h, weights_TK, ids_TK, counts_E)
            else:
                h = module(h)
        h.float().sum().backward()

    times = []
    with set_current_spmd_mesh(mesh):
        for e in experts:
            e.token_dispatcher.init_buffer()
        for i in range(args.warmup + args.iters):
            if i == args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            if i >= args.warmup:
                times.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated() / 2**30
        if args.profile:
            from torch.profiler import profile, ProfilerActivity

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    step()
                torch.cuda.synchronize()
            if rank == 0:
                table = None
                for key in ("self_cuda_time_total", "self_device_time_total"):
                    try:
                        table = prof.key_averages().table(sort_by=key, row_limit=80)
                        break
                    except Exception:
                        continue
                with open(args.profile, "w") as f:
                    f.write(table or "no table")
    stats = torch.tensor([statistics.median(times), peak], device=device)
    gathered = [torch.zeros_like(stats) for _ in range(size)]
    dist.all_gather(gathered, stats)
    if rank == 0:
        result = {
            "args": vars(args),
            "median_ms_per_rank": [g[0].item() for g in gathered],
            "peak_gib_per_rank": [g[1].item() for g in gathered],
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(json.dumps(result))
    experts[0].token_dispatcher.buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
