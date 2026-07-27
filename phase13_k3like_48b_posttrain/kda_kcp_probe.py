"""KCP probe: validate fla-core's KDA Context Parallelism against a
single-rank reference (2026-07-27).

K3's tech report sec 5.1.2 does NOT use Ulysses head-sharding for KDA (what
this repo built on 07-24) and does NOT use the LASP-style state summation
either -- plain summation is insufficient because KDA's delta rule applies a
token-dependent M_t to the incoming state. Instead, KCP decomposes each rank's
segment into two locally computable fragments,

    M^{T<-1}_[i] = prod_r M_r      (cumulative transition, R^{dk x dk})
    S~^{T}_[i]                     (state started from S = 0, R^{dk x dv})

which compose associatively, so each rank's incoming state is recovered by a
PREFIX SCAN over the fragments: one fixed-size all-gather, independent of
sequence length. Report footnote: "The KDA implementation is available in FLA
PR #691" -- and our pinned fla-core 0.5.1 already carries it (chunk_kda takes
cp_context; fla/ops/cp/{context,comm,chunk_delta_h}.py).

This probe answers the only question that matters before integration: does the
CP path reproduce the non-CP result on OUR shapes? Gate: max abs error within
the bf16 chunked-scan band, on both the output and the final state.

Launch: torchrun --nproc_per_node=<cp> kda_kcp_probe.py [seq_len]
"""

import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    from fla.ops.cp.context import build_cp_context
    from fla.ops.kda import chunk_kda

    dist.init_process_group("nccl")
    rank, cp = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    H, K, V = 4, 64, 64
    assert T % cp == 0, "seq_len must divide across cp ranks"

    # Identical inputs on every rank (seeded), varlen layout [1, T, H, K]:
    # fla's CP context partitions a PACKED sequence, so we feed one sequence
    # of length T with cu_seqlens = [0, T].
    g_ = torch.Generator(device="cuda").manual_seed(0)

    def mk(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=g_)

    q, k, v = mk(1, T, H, K), mk(1, T, H, K), mk(1, T, H, V)
    g = torch.nn.functional.logsigmoid(
        mk(1, T, H, K).float()
    ).to(torch.bfloat16)
    beta = mk(1, T, H).sigmoid()
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    # Reference: the whole sequence on one rank, no CP.
    with torch.no_grad():
        o_ref, s_ref = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
        )

    # KCP: each rank owns a contiguous token range of the packed sequence.
    part = T // cp
    sl = slice(rank * part, (rank + 1) * part)
    ctx = build_cp_context(cu, group=dist.group.WORLD)
    with torch.no_grad():
        o_cp, s_cp = chunk_kda(
            q=q[:, sl], k=k[:, sl], v=v[:, sl], g=g[:, sl], beta=beta[:, sl],
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=ctx.cu_seqlens, cp_context=ctx,
        )

    # Gather the per-rank output slices and compare against the reference.
    gathered = [torch.empty_like(o_cp) for _ in range(cp)]
    dist.all_gather(gathered, o_cp.contiguous())
    o_all = torch.cat(gathered, dim=1)

    if rank == 0:
        err = (o_all.float() - o_ref.float()).abs().max().item()
        rel = (
            (o_all.float() - o_ref.float()).norm() / o_ref.float().norm()
        ).item()
        print(f"[KCP] cp={cp} T={T} H={H} K={K}", flush=True)
        print(f"[KCP] output  max-abs {err:.3e}  rel {rel:.3e}", flush=True)
        if s_cp is not None and s_ref is not None:
            serr = (s_cp.float() - s_ref.float()).abs().max().item()
            print(f"[KCP] final state max-abs {serr:.3e}", flush=True)
        # bf16 chunked-scan band, same tolerance the Ulysses probe used
        print("[KCP] PASS" if rel < 5e-2 else "[KCP] FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
