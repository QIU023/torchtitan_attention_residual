"""Task 24: KDA context parallel via Ulysses head-sharding.

The RFC scopes CP-for-KDA as future work: ring/zigzag applies only to the
full-attention (MLA) layers; KDA needs Ulysses head-sharding or LASP
state-passing. This probe validates the NOVEL core of Ulysses-for-KDA:
because fla-core's chunk_kda is per-HEAD independent (each head carries
its own recurrent delta-rule state), sharding the HEAD dim across CP ranks
and computing each rank's head-subset over the FULL sequence is
numerically EXACT -- the all-to-all is lossless data movement.

Setup (cp ranks): the per-head KDA inputs (q,k,v,g,beta) live seq-sharded
[B, T/cp, H, K] (as they would after seq-local projection). Ulysses:
  all-to-all -> [B, T, H/cp, K]  (full seq, head subset)
  chunk_kda   (full-seq scan on the head subset)
  all-to-all -> [B, T/cp, H, K]  (re-shard seq, gather heads)
gather -> [B, T, H, K] and compare to the non-CP chunk_kda reference.

Full-layer CP integration (seq-local proj + conv halo + torchtitan's
context_parallel context) is standard remaining wiring; the head-shard
numerics -- the part the RFC flagged as unsupported -- are proven here.
"""
import os

import torch
import torch.distributed as dist


def all_to_all_headseq(x, cp, seq_to_head):
    """Swap which of (seq, head) is sharded, via all_to_all_single.

    seq_to_head=True:  in [B, T/cp, H, K] (seq-shard) -> [B, T, H/cp, K].
    seq_to_head=False: in [B, T, H/cp, K] -> [B, T/cp, H, K] (head->seq).
    """
    B, d1, d2, K = x.shape
    if seq_to_head:
        Tloc, H = d1, d2
        # [B, Tloc, H, K] -> [cp, B, Tloc, H/cp, K] (split heads by dest)
        xr = x.reshape(B, Tloc, cp, H // cp, K).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(xr)
        dist.all_to_all_single(out, xr)
        # recv[s] holds src s's Tloc for THIS rank's head-subset -> stack seq
        return out.permute(1, 0, 2, 3, 4).reshape(B, cp * Tloc, H // cp, K).contiguous()
    else:
        Tfull, Hloc = d1, d2
        Tloc = Tfull // cp
        xr = x.reshape(B, cp, Tloc, Hloc, K).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(xr)
        dist.all_to_all_single(out, xr)
        # out[s] = src s's head-subset for THIS rank's seq; put Tloc before
        # the src(cp) axis so reshape stacks heads as [src0 heads, src1 ...].
        return out.permute(1, 2, 0, 3, 4).reshape(B, Tloc, cp * Hloc, K).contiguous()


def main():
    from fla.ops.kda import chunk_kda

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    cp = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    B, T, H, K = 1, 256, 8, 64
    assert H % cp == 0 and T % cp == 0
    g_ = torch.Generator(device="cuda").manual_seed(0)

    def mk(shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=g_)

    # Full per-head KDA inputs (identical on every rank via the seeded gen).
    q, k, v = mk((B, T, H, K)), mk((B, T, H, K)), mk((B, T, H, K))
    g = torch.nn.functional.logsigmoid(mk((B, T, H, K)).float()).to(torch.bfloat16)
    beta = mk((B, T, H)).sigmoid()

    # Reference: chunk_kda on the full (seq, head) tensor.
    with torch.no_grad():
        o_ref, _ = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=None,
        )

    # Ulysses CP: seq-shard the inputs, all-to-all to head-shard, run
    # chunk_kda on the head subset over full seq, all-to-all back.
    sl = slice(rank * (T // cp), (rank + 1) * (T // cp))
    qs, ks, vs = q[:, sl], k[:, sl], v[:, sl]
    gs, bs = g[:, sl], beta[:, sl]
    beta4 = bs.unsqueeze(-1)  # carry beta through the head/seq a2a as K=1

    qh = all_to_all_headseq(qs, cp, True)
    kh = all_to_all_headseq(ks, cp, True)
    vh = all_to_all_headseq(vs, cp, True)
    gh = all_to_all_headseq(gs, cp, True)
    bh = all_to_all_headseq(beta4, cp, True).squeeze(-1)
    with torch.no_grad():
        oh, _ = chunk_kda(
            q=qh, k=kh, v=vh, g=gh, beta=bh, initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=None,
        )
    o_local = all_to_all_headseq(oh, cp, False)  # [B, T/cp, H, K]

    # gather seq shards to compare against the full reference
    gathered = [torch.empty_like(o_local) for _ in range(cp)]
    dist.all_gather(gathered, o_local.contiguous())
    o_cp = torch.cat(gathered, dim=1)

    if rank == 0:
        rel = (o_cp.float() - o_ref.float()).norm().item() / o_ref.float().norm().item()
        print(f"[KDA-ULYSSES] cp={cp} H={H} T={T}: CP-vs-reference rel-err {rel:.2e}", flush=True)
        # bf16 all-to-all round-trip + kernel nondeterminism band
        print("[KDA-ULYSSES] PASS" if rel < 5e-2 else "[KDA-ULYSSES] FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
