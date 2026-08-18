# PR #24 — Kimi K3: context parallelism (KCP for KDA, Ulysses for MLA)

**Target**: `pytorch/torchtitan`, on top of #4025 and PR22
**Scope**: `apply_cp_kimi_k3` in `parallelize.py`, `kcp.py` (the KDA sequence-parallel path),
`vit_cp_plan.py` plus the dynamic-CP execution half in `multimodal_model.py` / `moonvit.py`

## What this is, precisely

Two mechanisms on disjoint layer kinds, both active in one CP run -- not a choice
between them:

* **KDA layers: KCP** (tech report sec 5.1.2). The sequence stays sharded end to end. A
  prefix scan over (cumulative transition, zero-started state) fragments recovers each
  rank's true incoming recurrent state in one fixed-size all-gather, and a fixed-size halo
  covers the short convolutions. No rank ever materializes the full sequence.
* **MLA layers: Ulysses head-sharding.** One fused differentiable all-to-all on the cp
  sub-mesh; each rank runs SDPA on its head subset over the full sequence.

Saying this plainly matters because the two are easy to conflate. KCP decomposes the
delta-rule recurrence and says nothing about softmax attention, so it does not replace
Ulysses -- it applies where Ulysses cannot.

**`kda_cp_mode` defaults to `"kcp"`.** Ulysses for the KDA layers is kept as the A/B and is
not what K3 does: it gives every rank the whole sequence for its head subset, so activation
memory does not fall with `cp` and the context length K3 targets is out of reach. An earlier
draft of this PR described Ulysses-for-KDA as the mechanism, which was accurate when the
default was Ulysses and is no longer.

## Why not the upstream dispatcher

`apply_cp_to_forward` dispatches on torchtitan's SDPA type and applies ring
attention. Neither fits:

- **KDA cannot ring.** fla-core's `chunk_kda` is a sequential scan over the
  sequence; there is no ring formulation of it.
- **MLA's `inner_attention` is not the SDPA type the dispatcher recognises.**

`apply_cp_to_forward`'s own TODO says it is a temporary workaround and that CP
redistribution should eventually be declarative via `ShardingConfig`. Ulysses would fit
that -- it is one all-to-all, i.e. a `Shard(seq) -> Shard(head)` pair on the cp axis. KCP
would not: the sequence stays sharded and the recurrence is recomputed from prefix-scanned
fragments, so no placement of the module's tensors describes it. This PR therefore matches
the shape of upstream CP as it exists today and should move when that TODO does.

So CP is wired module-internally: `KimiMLAAttention` and `KimiDeltaAttention`
each get a `_cp_group` and a `._forward_cp`. The MLA branch validates
`num_heads % (tp * cp) == 0`, since under TP the head axis is already tp-sharded
and Ulysses splits the local heads further.

## The KDA path has two cross-rank dependencies, with different shapes

The recurrence and the convolution need different things, which is why KCP is not just a
halo:

* **the recurrence** needs each rank's true incoming state, and that does NOT decompose by
  summation -- the delta rule applies a token-dependent transition, so LASP-style state
  summation is wrong. fla's `chunk_kda(cp_context=...)` does the prefix scan;
* **the convolution** needs only the previous rank's tail, its support being finite -- one
  fixed-size exchange, no scan.

## The halo comes from fla rather than from us

KDA's short convolution has a receptive field, so a rank holding a sequence shard
still needs the previous rank's tail at chunk edges. `kcp.py` wires fla's
`causal_conv1d_cp` for that -- a real `autograd.Function` that exchanges the tail in
the forward and the matching `dx` in the backward. Verified bit-exact against a
single-rank reference at cp2, cp4 and cp8, with a control run at 7.2e-02 to show the
test can fail.

**This paragraph used to say "`kcp.py` implements the halo exchange", and that was
stale.** It did once, and the way it was wrong is the reason this section exists: the
hand-rolled version fetched the neighbour's conv state with `dist.all_gather`, whose
forward was bit-exact but which is not autograd-aware, so the gradient each rank owed
its left neighbour's tail was silently dropped. Rank 0's last `W - 1` tokens came out
~60% wrong while every interior token stayed exact. An error confined to `W - 1`
boundary tokens per rank does not move a loss curve, which is why a forward-only
bit-exactness check passed it -- and why the fix was to use the library function that
carries a backward rather than to patch the halo.

`chunk_kda` itself is bit-exactly per-head independent -- verified against a
single-rank reference -- which is what makes head sharding exact rather than
approximate.

## Evidence

The KDA path is checked against a single-rank reference, forward AND backward, because a
wrong gradient still lets the loss fall -- no training cell can see it
(`matrix_scripts/kcp_batch_parity.py`). Rank r's output must equal the full-sequence
forward's slice r, per batch row:

    cp=2   forward 4.310e-03 (rank 1) / 4.660e-03 (rank 0)
           gradients 14 compared, worst k_conv1d.weight 8.929e-03, none missing
    cp=4   worst gradient o_norm.weight    7.752e-03
    cp=8   worst gradient f_b_proj.weight  1.364e-02

All bf16 noise. **cp>=4 is the load-bearing case**: it is the first configuration with a
MIDDLE rank, so the prefix scan composes more than two fragments. A probe at cp=2 alone
would pass with the scan's composition rule broken.

The worst gradient being a conv weight is the point -- the halo is where a hand-rolled
`all_gather` once dropped the gradient owed to the left neighbour while the forward stayed
exact.

Batch axis: fla's `causal_conv1d_cp` asserts `[1, T, D]`, so the CP path loops the batch.
Flattening into one packed sequence would be wrong rather than awkward --
`build_cp_context` cuts the GLOBAL packed sequence into contiguous rank-ordered pieces while
a rank holds piece r of EVERY sequence, so the layouts coincide only at B = 1. The loop is
also what the recurrence needs, since delta-rule state must not carry between sequences.

In the training matrix, 7 CP cells across 3 arms (text / multimodal / multimodal+LoRA), all
passing with `kda_cp_mode=kcp` on every rank that holds KDA layers:

    Applied CP cp_degree=2: 4 MLA layer(s) Ulysses, 9 KDA layer(s) kda_cp_mode=kcp

## Honest gap

The MLA layers are head-parallel, not sequence-parallel. KCP gives the KDA layers a sharded
sequence end to end, but a ring/zigzag formulation for softmax attention is separate work
and is not here. (An earlier draft of this section said sequence-dimension CP was not
implemented at all, which was true when Ulysses was the only path and is not now.)
