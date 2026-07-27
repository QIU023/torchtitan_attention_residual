"""KCP backward check.

The FLA #691 author states outputs AND GRADIENTS are lossy vs all-to-all CP.
Our forward probe measured bit-exact -- but it ran under no_grad, so the
backward was never exercised. This tests the gradients.
"""

import os
import sys

import torch
import torch.distributed as dist

from fla.ops.cp.context import build_cp_context
from fla.ops.kda import chunk_kda


def main() -> None:
    dist.init_process_group("nccl")
    rank, cp = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    H, K = 4, 64
    gen = torch.Generator(device="cuda").manual_seed(0)

    def rnd(*shape):
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, generator=gen
        )

    q, k, v = rnd(1, T, H, K), rnd(1, T, H, K), rnd(1, T, H, K)
    g = (
        torch.nn.functional.logsigmoid(rnd(1, T, H, K).float())
        .to(torch.bfloat16)
    )
    beta = rnd(1, T, H).sigmoid()
    for t in (q, k, v, g, beta):
        t.requires_grad_(True)
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    gy = rnd(1, T, H, K)

    names = ("q", "k", "v", "g", "beta")
    tensors = (q, k, v, g, beta)

    # reference: whole sequence on one rank
    o_ref, _ = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
    )
    o_ref.backward(gy)
    ref = {n: t.grad.detach().clone() for n, t in zip(names, tensors)}
    for t in tensors:
        t.grad = None

    # KCP: local shard, sequence stays sharded
    part = T // cp
    sl = slice(rank * part, (rank + 1) * part)
    ctx = build_cp_context(cu, group=dist.group.WORLD)
    o_cp, _ = chunk_kda(
        q=q[:, sl], k=k[:, sl], v=v[:, sl], g=g[:, sl], beta=beta[:, sl],
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=ctx.cu_seqlens, cp_context=ctx,
    )
    o_cp.backward(gy[:, sl])

    if rank == 0:
        print(f"[KCP-BWD] cp={cp} T={T}", flush=True)
        for n, t in zip(names, tensors):
            a = t.grad[:, sl].float()
            b = ref[n][:, sl].float()
            rel = ((a - b).norm() / b.norm().clamp(min=1e-12)).item()
            mx = (a - b).abs().max().item()
            print(f"[KCP-BWD] d{n:<5} rel {rel:.3e}  max-abs {mx:.3e}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
