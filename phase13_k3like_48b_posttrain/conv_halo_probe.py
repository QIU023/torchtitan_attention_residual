"""The KCP blocker: can KDA's short convolution be made CP-correct?

KCP (report sec 5.1.2) shards the sequence across ranks. The chunkwise delta
rule composes via a prefix scan over fragments, which fla-core 0.5.1 already
implements and this repo already validated bit-exact both directions
(kda_kcp_probe.py). What was left open is the SHORT CONVOLUTION in front of
q/k/v: it is causal with support W = short_conv_kernel_size, so rank r's first
W-1 outputs depend on the tail of rank r-1's input. Shard naively and those
outputs are computed against zero padding instead -- wrong, and wrong in a way
that shrinks as sequence length grows, so it hides in an averaged loss.

fla's convolution module has no CP support (no cp_context, no halo). But
ShortConvolution.forward already takes `cache`, an [N, D, W] left-context state
used for incremental decoding, and can return its own final state. That is
exactly a halo: one fixed-size point-to-point exchange per rank, independent of
sequence length, and with no dependency chain because the support is finite --
rank r needs only rank r-1, not a scan.

This probe checks that the halo reproduces the unsharded convolution, and
includes the control that makes the check meaningful: the same sharding WITHOUT
the halo must be visibly wrong, otherwise the test has no power.

Launch: torchrun --nproc_per_node=<cp> conv_halo_probe.py [seq_len]
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def exchange_halo(state: torch.Tensor, rank: int, world: int, group=None):
    """Shift each rank's conv final state one rank to the right.

    Rank 0 gets None (a true sequence start, so zero left padding is correct);
    rank r gets rank r-1's state. Implemented as an all_gather of a tensor whose
    size is [N, D, W] -- independent of sequence length -- because a
    send/recv pair would need careful ordering to avoid deadlock and the payload
    here is tiny either way.
    """
    gathered = [torch.empty_like(state) for _ in range(world)]
    dist.all_gather(gathered, state.contiguous(), group=group)
    return None if rank == 0 else gathered[rank - 1]


def main() -> None:
    from fla.modules import ShortConvolution

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    D, W = 256, 4
    assert T % world == 0

    torch.manual_seed(0)
    conv = ShortConvolution(hidden_size=D, kernel_size=W, activation="silu").cuda()
    conv = conv.to(torch.bfloat16)
    # identical weights on every rank
    for p in conv.parameters():
        dist.broadcast(p.data, src=0)

    g = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn(1, T, D, device="cuda", dtype=torch.bfloat16, generator=g)

    with torch.no_grad():
        ref, _ = conv(x, cache=None, output_final_state=False)

    part = T // world
    sl = slice(rank * part, (rank + 1) * part)
    x_local = x[:, sl].contiguous()

    # Every rank computes the state its LEFT NEIGHBOUR needs, then shifts.
    with torch.no_grad():
        _, my_state = conv(x_local, cache=None, output_final_state=True)
    halo = exchange_halo(my_state, rank, world)

    with torch.no_grad():
        # cache is updated in place by fla, so hand over a copy: the same
        # buffer is the neighbour's state and would be clobbered.
        y_halo, _ = conv(
            x_local,
            cache=None if halo is None else halo.clone(),
            output_final_state=False,
        )
        # CONTROL: no halo, so every rank starts from zero left padding.
        y_nohalo, _ = conv(x_local, cache=None, output_final_state=False)

    def gather_full(y):
        buf = [torch.empty_like(y) for _ in range(world)]
        dist.all_gather(buf, y.contiguous())
        return torch.cat(buf, dim=1)

    full_halo = gather_full(y_halo)
    full_nohalo = gather_full(y_nohalo)

    if rank == 0:
        r_halo = (
            (full_halo.float() - ref.float()).norm() / ref.float().norm()
        ).item()
        r_nohalo = (
            (full_nohalo.float() - ref.float()).norm() / ref.float().norm()
        ).item()
        max_halo = (full_halo.float() - ref.float()).abs().max().item()
        print(f"[CONV-CP] cp={world} T={T} D={D} W={W}", flush=True)
        print(f"[CONV-CP] with halo    rel {r_halo:.3e}  max-abs {max_halo:.3e}",
              flush=True)
        print(f"[CONV-CP] control      rel {r_nohalo:.3e}  (no halo)", flush=True)
        # The error a missing halo introduces is confined to W-1 tokens per
        # rank boundary, so report it where it actually lives -- a
        # sequence-averaged norm dilutes it as T grows, which is exactly why
        # this bug would survive a loss-curve comparison.
        b = part
        boundary = slice(b, b + W - 1)
        r_boundary = (
            (full_nohalo[:, boundary].float() - ref[:, boundary].float()).norm()
            / ref[:, boundary].float().norm()
        ).item()
        print(
            f"[CONV-CP] control at the first boundary ({W-1} tokens): "
            f"rel {r_boundary:.3e}",
            flush=True,
        )
        ok = r_halo < 1e-6 and r_boundary > 1e-2
        print("[CONV-CP] PASS" if ok else "[CONV-CP] FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
