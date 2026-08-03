# PR #24 — Kimi K3: context parallelism (Ulysses head-sharding)

**Target**: `pytorch/torchtitan`, on top of #4025 and PR22
**Scope**: the CP wiring in `parallelize.py` plus `kcp.py` (KDA conv halo)

## What this is, precisely

**Ulysses head-parallelism**, not ring/zigzag sequence sharding. Each rank runs
conv, scan and SDPA on its own subset of HEADS over the FULL sequence, with one
fused differentiable all-to-all on the cp sub-mesh.

Saying this plainly matters, because "CP support" usually means sequence
sharding and this is not that.

## Why not the upstream dispatcher

`apply_cp_to_forward` dispatches on torchtitan's SDPA type and applies ring
attention. Neither fits:

- **KDA cannot ring.** fla-core's `chunk_kda` is a sequential scan over the
  sequence; there is no ring formulation of it.
- **MLA's `inner_attention` is not the SDPA type the dispatcher recognises.**

So CP is wired module-internally: `KimiMLAAttention` and `KimiDeltaAttention`
each get a `_cp_group` and a `._forward_cp`. The MLA branch validates
`num_heads % (tp * cp) == 0`, since under TP the head axis is already tp-sharded
and Ulysses splits the local heads further.

## KDA needs a halo, and it is exact

KDA's short convolution has a receptive field, so a rank holding a head subset
still needs neighbouring positions at chunk edges. `kcp.py` implements the halo
exchange. Verified bit-exact against a single-rank reference at cp2, cp4 and
cp8, with a control run at 7.2e-02 to show the test can fail.

`chunk_kda` itself is bit-exactly per-head independent -- verified against a
single-rank reference -- which is what makes head sharding exact rather than
approximate.

## Evidence

    cp2 alone                    trains, loss matches the matrix
    fsdp2 x cp2                  trains
    tp2 x cp2                    8-step loss curve, max |dloss| 0.00824
    tp2 x pp2 x cp2                                            0.01007
    fsdp2 x tp2 x cp2            trains
    ep2 x fsdp2 x tp2 x cp2      trains

## Honest gap

There is no `apply_cp` function; the wiring is inline in the main entry
(parallelize.py:140-205). That is a structural inconsistency with the other
axes, each of which has an `apply_*`, and it should be factored before merge.

Sequence-dimension CP for the MLA layers is NOT implemented. If a reviewer wants
ring/zigzag over the sequence, that is separate work; what is here is head
parallelism, verified as such.
