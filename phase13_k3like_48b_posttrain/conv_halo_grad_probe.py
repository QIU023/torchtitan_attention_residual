# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Does conv_with_halo carry gradients across the CP rank boundary?

The forward halo was measured bit-exact, but the halo is fetched with
dist.all_gather, which is not autograd-aware. If that is the case, the gradient
that rank r owes rank r-1's tail tokens is silently dropped, and the error is
confined to W-1 boundary tokens per rank -- invisible in a loss curve.

Reference: run the unsharded conv on the full sequence and slice each rank's
gradient out of it. Compare against (a) conv_with_halo and (b) fla's own
causal_conv1d_cp, which ships a real autograd.Function for this.

torchrun --nproc_per_node=2 conv_halo_grad_probe.py
"""

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    from fla.modules.convolution import ShortConvolution

    torch.manual_seed(0)
    D, W, T_TOTAL = 32, 4, 64
    t_loc = T_TOTAL // world

    conv = ShortConvolution(D, W, activation="silu").to(dev).to(torch.float32)
    # every rank must hold identical weights; seed alone is not enough once the
    # module has been moved, so broadcast.
    for p in conv.parameters():
        dist.broadcast(p.data, src=0)

    x_full = torch.randn(1, T_TOTAL, D, device=dev, dtype=torch.float32)
    g_full = torch.randn(1, T_TOTAL, D, device=dev, dtype=torch.float32)
    dist.broadcast(x_full, src=0)
    dist.broadcast(g_full, src=0)

    # ---- reference: unsharded, then slice ----
    xr = x_full.clone().requires_grad_(True)
    out_ref, _ = conv(xr, cache=None, output_final_state=False)
    out_ref.backward(g_full)
    ref_out = out_ref.detach()[:, rank * t_loc : (rank + 1) * t_loc]
    ref_grad = xr.grad.detach()[:, rank * t_loc : (rank + 1) * t_loc]
    conv.zero_grad(set_to_none=True)

    # ---- ours ----
    from torchtitan.models.kimi_k3.kcp import build_kcp_context, conv_with_halo

    xl = x_full[:, rank * t_loc : (rank + 1) * t_loc].clone().requires_grad_(True)
    gl = g_full[:, rank * t_loc : (rank + 1) * t_loc]
    ctx = build_kcp_context(t_loc, dist.group.WORLD, dev, conv1d_kernel_size=W)
    out_ours = conv_with_halo(conv, xl, ctx)
    out_ours.backward(gl)

    def rel(a, b):
        d = (a - b).abs().max().item()
        s = b.abs().max().item()
        return d / s if s else d

    fwd_err = rel(out_ours.detach(), ref_out)
    grad_err = rel(xl.grad.detach(), ref_grad)

    # where does the gradient error live? the halo only touches the FIRST W-1
    # tokens of a non-zero rank, but the gradient a rank owes its neighbour
    # lands on its LAST W-1 tokens.
    tail = slice(t_loc - (W - 1), t_loc)
    tail_err = rel(xl.grad.detach()[:, tail], ref_grad[:, tail])
    interior_err = rel(
        xl.grad.detach()[:, : t_loc - (W - 1)], ref_grad[:, : t_loc - (W - 1)]
    )

    for r in range(world):
        if r == rank:
            print(
                f"[rank {rank}] fwd_rel={fwd_err:.3e}  grad_rel={grad_err:.3e}  "
                f"grad_tail(last {W-1})={tail_err:.3e}  grad_interior={interior_err:.3e}",
                flush=True,
            )
        dist.barrier()

    if rank == 0:
        print(
            "\ngrad_tail at roundoff (~1e-8 fp32) == the cross-rank gradient "
            "arrives. A grad_tail near 1e-1 with grad_interior exactly 0 is the "
            "signature of a non-autograd-aware halo exchange.",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    main()
