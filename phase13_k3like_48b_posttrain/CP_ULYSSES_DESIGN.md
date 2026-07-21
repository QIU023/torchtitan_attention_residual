# KDA context-parallel via Ulysses — design (2026-07-21)

Turns the RFC's "CP out of scope" into a specced, numerically-validated
plan. The novel/risky numerics are DONE (bit-exact); what remains is
standard-but-nontrivial wiring. Deliberately NOT landed as a rush before
the RFC review -- the guard in `parallelize.py` stays until this is
implemented + parity-tested, so CP fails loudly rather than silently
wrong.

## What is proven (kda_ulysses_cp_probe.py)

`chunk_kda` is **bit-exactly per-head independent** (full[:h] vs
head-subset rel-err 0.00 -- each head carries its own delta-rule state).
Therefore Ulysses head-sharding is numerically EXACT: seq-shard ->
all-to-all(seq<->head) -> chunk_kda on the head subset over the FULL
sequence -> all-to-all back reconstructs the non-CP reference with
**rel-err 0.00 at cp=2 and cp=4**. This is the "per-layer all-gather at
the KDA boundary" the parallelize.py guard calls non-trivial -- validated.

## KDA layer forward under CP (the wrapper to add)

Input arrives seq-sharded `[B, T/cp, D]` (torchtitan CP context shards
the sequence). Order matters because the short conv and the scan both
need the full sequence:

1. Seq-local projections (linear, no cross-seq): q/k/v_proj, f_a/f_b
   (gate), b_proj (beta), g_a/g_b (out-gate) -> per-head `[B, T/cp, H, *]`.
2. **all-to-all** each -> `[B, T, H/cp, *]` (full seq, head subset).
3. Short causal conv on q/k/v: apply on the full seq with the conv weight
   SLICED to the head-subset channels `[r*(H/cp)*K : (r+1)*(H/cp)*K]`
   (conv is per-channel, so slicing is exact). No halo needed because the
   a2a already gathered the full sequence.
4. `fused_kda_gate(g, A_log[hs], dt_bias[hs])` and `chunk_kda(q,k,v,g,beta)`
   on the head subset (A_log/dt_bias/beta sliced to the subset's heads).
5. `o_norm(o, g_out)` (per head_dim, head-local).
6. **all-to-all** back -> `[B, T/cp, H, K]`; reshape; `o_proj` (mixes
   heads, now gathered) -> `[B, T/cp, D]`.

All-to-all helper is `all_to_all_headseq` from the probe (validated
round-trip). Autograd: all_to_all_single is differentiable; the backward
is the transposed all-to-all, so grads flow.

## MLA layers under CP

MLA (full attention) composes with CP via the standard torchtitan path
(`apply_cp_to_attention_module` + the SDPA ring/zigzag dispatcher) -- no
new work, MLA is not the blocker. The hybrid model just needs both:
Ulysses for KDA layers, ring for MLA layers, on the same cp mesh (both
keep the sequence seq-sharded at layer boundaries, so no extra gather
between layers).

## parallelize.py wiring

Replace the `NotImplementedError` (line ~155) with: register the cp mesh,
route KDA layers through the Ulysses wrapper (steps above) and MLA layers
through `apply_cp_to_attention_module`. The AttnRes cross-block skip edges
are position-wise -> they stay seq-sharded and need no CP-specific
handling (same as the residual stream).

## Verification plan (once wired)

- Layer parity: KDA layer under cp=2 vs cp=1, rel-err within bf16 band.
- Full-model parity: `torchtitan.train` cp=2 vs cp=1 loss (same-init via
  checkpoint), and **CP composed with FSDP/PP/EP** (CP+FSDP, CP+PP,
  PP+CP+FSDP) -- the mixes the guard currently blocks.

## Why not landed now

The numerics (the part that could be silently wrong) are proven. The
wiring touches the KDA forward, the conv-weight slicing, and the CP
context -- multi-hour with real risk of a subtly-wrong hybrid CP path.
Landing it unverified right before the RFC review would be worse than the
honest guard. Scoped as the next focused block (or an H200 leg alongside
the 1M-context work CP enables).
