# KDA/MLA context-parallel: real Ulysses design + full-5D requirement (updated 2026-07-23)

> Supersedes the 2026-07-21 draft below the line. That version predates the
> CP that actually landed (`f4b6f46f`, `ec417b21`, `48285050`) and described
> a plan that was never fully built as specced (MLA did NOT get
> `apply_cp_to_attention_module` ring/zigzag; both KDA and MLA got the same
> all-gather + head-shard hybrid instead). This revision documents what is
> ACTUALLY landed, the two problems it has (verified by reading the code,
> not the docs), and requires the fix to land as full 5D (FSDP x TP x CP x
> EP x PP), not single-axis CP.

## What is actually landed today (verified against model.py, not docs)

`KimiMLAAttention.forward` (model.py ~L343) and `KimiDeltaAttention.forward`
(model.py ~L601) both do, identically:

```
x: [B, T/cp, D]  (seq-sharded input)
  -> dist_nn.all_gather(x, group=cp_group)          # FULL sequence, every rank
x: [B, T_full, D]
  -> q_proj / kv_a_proj / kv_b_proj / conv / f_a,b / b_proj / g_a,b   # on T_full, REDUNDANT per rank
  -> LOCAL slice to this rank's H/cp heads (no comm -- data is already full)
  -> chunk_kda / SDPA on H/cp heads over T_full       # the only op that shrinks by cp
  -> dist_nn.all_gather(attn_out, group=cp_group)     # heads back together
  -> o_proj on T_full                                  # REDUNDANT per rank again
  -> slice back to this rank's [B, T/cp, D]
```

## Problem 1 (real, not cosmetic): this is Sequence Parallelism, not Ulysses -- no memory scaling

The all-gather happens BEFORE any projection. Every rank materializes
`[B, T_full, D]` for x, and then q/k/v/conv/gate activations at `T_full`,
regardless of cp degree. This is architecturally Megatron Sequence
Parallelism (all-gather at the block boundary, full computation, slice back)
with a head-split bolted onto the one inner op (chunk_kda / SDPA) purely to
avoid redundant FLOPs there -- it is NOT Ulysses (no all-to-all, no
avoidance of full-sequence materialization) and NOT ring attention for MLA
either (ring never materializes full K/V on one rank at all).

**Consequence for 1M-context (the actual K3 target): activation memory at
the all-gather is O(T_full x D) per rank, independent of cp.** A 1M-token
sequence OOMs at this line at ANY cp degree -- CP as landed buys
correctness + a compute discount on one sub-op, not the memory scaling that
is the entire point of CP for extreme length. This is not "an optimization
for later"; it is the blocker for the stated 1M-context goal. Real Ulysses
(input arrives seq-sharded, project LOCALLY at T/cp, THEN all-to-all
seq<->head so no rank ever holds T_full x D) is a requirement, not a
nice-to-have -- see the wiring spec below (kept from the 07-21 draft, still
valid and still only numerically probed, not landed).

## Problem 2 (a real bug, not just a gap): CP silently no-ops under TP

Verified in the code, not just the docstring: `parallelize.py` has two
independent, unguarded blocks --

```python
if parallel_dims.tp_enabled: ...   # L100
if parallel_dims.cp_enabled: ...   # L135
```

no check between them. If both flags are set: TP wraps the model so
activations arrive as `DTensor` at layer boundaries; separately, CP sets
`_cp_group` on every KDA/MLA module. At forward time, the CP block's own
guard (`model.py` L354 MLA / equivalent in KDA) is:

```python
if cp_group is not None and not isinstance(x, DTensor) and dist.get_world_size(cp_group) > 1:
```

`isinstance(x, DTensor)` is **False** only when TP is off. Under TP, this
condition is false, so **the entire CP block is silently skipped at every
forward call** -- no error, no warning, training proceeds and produces a
plausible loss curve with CP configured-but-inert. A user who sets
`tp_degree>1, cp_degree>1` believes they have both; they get TP only, no
error to tell them otherwise.

**Action before ANY further CP work**: make this fail loudly (raise
`NotImplementedError` at `parallelize.py` build time if `cp_enabled and
tp_enabled`, not a runtime no-op) until real CP+TP lands. This is a
same-day fix independent of the Ulysses rewrite; do it first.

## Target: full 5D (FSDP x TP x CP x EP x PP), not single-axis CP

FSDP, EP, and PP are ALREADY proven to compose with the current (SP-style)
CP -- this is real evidence from the 8-card matrix, not an assumption:

| combo | result | source |
|---|---|---|
| CP+FSDP (dp4,cp2) | loss 6.751, trains | EIGHTCARD_VERIFICATION_2026-07-21.md |
| PP+CP+FSDP (dp2,cp2,pp2) | loss 7.059, trains | same |
| CP+EP (dp4,cp2,ep2) | loss 6.751, trains | same |
| CP4+FSDP (dp2,cp4) | loss 6.891, trains | same |
| full-param FSDP+CP+EP+PP (dp2,cp2,ep2,pp2) | PASS, loss 7.059 | same |
| LoRA FSDP+CP+EP+PP | PASS, loss 7.644 | same |

**TP is confirmed the only remaining axis** -- not a guess, the silent-no-op
in Problem 2 is the concrete mechanism, and no other axis has an equivalent
gate. Because FSDP/EP/PP compose by leaving the attention module's external
contract untouched (seq-sharded plain tensor in, seq-sharded plain tensor
out -- CP's all-gather/slice happens entirely INSIDE the module and restores
the boundary before returning), swapping the internals from all-gather to
real Ulysses should preserve that contract and FSDP/EP/PP composition
should carry over -- but this is a "should", not proven, and must be
re-verified after the swap, not assumed.

**Full 5D means the deliverable is not "CP works" but "CP+TP works, and the
full FSDP x TP x CP x EP x PP matrix (or the largest sub-mesh that fits the
card count) is verified", matching the rigor already applied to the
LoRA-x-parallelism and full-param matrices.**

## CP+TP: what real support requires (new work, not in the 07-21 draft)

Mathematically CP+TP is standard (Megatron-Core, DeepSpeed-Ulysses, veScale
all support it) -- the blocker is this fork's engineering, not the math.
Concretely:

1. **2D device mesh `(tp, cp)`**, with CP-only sub-process-groups derived
   correctly so collectives never cross TP rank boundaries incorrectly (and
   vice versa for TP's own collectives).
2. **Head-count divisibility**: effective head-groups = TP_degree x
   CP_degree once both shard the head dimension. Check MLA's `num_heads`
   and KDA's `num_heads` against the target TP x CP products -- both are
   modest counts and may bind before larger meshes.
3. **DTensor-vs-plain-tensor reconciliation**: the fork's boundary
   discipline (plain tensor at every module boundary, for fla-core triton
   kernels / PP P2P / `block_attn_res` stacking) means CP's collectives
   cannot rely on automatic DTensor multi-axis dispatch. Either (a) make
   the Ulysses all-to-all explicitly operate on the CP sub-mesh while a
   TP-sharded feature dimension passes through untouched (plain-tensor,
   manual bookkeeping, consistent with how TP already handles this
   elsewhere in the file), or (b) a narrower interim: TP wraps the
   projections only, CP's all-to-all still operates on plain tensors by
   converting DTensor<->local at the CP boundary (mirrors the existing
   `_to_local_if_dtensor` pattern KDA already uses for TP+fla-core).
4. **Order matters**: do Ulysses-CP-alone first (Problem 1's fix). Verify
   it alone (rel-err parity vs cp=1, matches the already-proven
   kda_ulysses_cp_probe.py numerics but now wired through
   `torchtitan.train`) before adding TP into the mesh -- do not attempt
   CP+TP on top of a still-unverified single-axis Ulysses rewrite.

## KDA layer forward under real Ulysses CP (kept from 07-21, still the plan)

Input arrives seq-sharded `[B, T/cp, D]`. Order matters because the short
conv and the scan both need the full sequence:

1. Seq-local projections (linear, no cross-seq): q/k/v_proj, f_a/f_b
   (gate), b_proj (beta), g_a/g_b (out-gate) -> per-head `[B, T/cp, H, *]`.
2. **all-to-all** each -> `[B, T, H/cp, *]` (full seq, head subset).
3. Short causal conv on q/k/v: apply on the full seq with the conv weight
   SLICED to the head-subset channels `[r*(H/cp)*K : (r+1)*(H/cp)*K]`
   (conv is per-channel, so slicing is exact). No halo needed because the
   a2a already gathered the full sequence for this head subset.
4. `fused_kda_gate(g, A_log[hs], dt_bias[hs])` and `chunk_kda(q,k,v,g,beta)`
   on the head subset (A_log/dt_bias/beta sliced to the subset's heads).
5. `o_norm(o, g_out)` (per head_dim, head-local).
6. **all-to-all** back -> `[B, T/cp, H, K]`; reshape; `o_proj` (mixes
   heads, now gathered) -> `[B, T/cp, D]`.

All-to-all helper is `all_to_all_headseq` from `kda_ulysses_cp_probe.py`
(validated round-trip, rel-err 0.00 at cp=2/cp=4). Autograd:
`all_to_all_single` is differentiable; the backward is the transposed
all-to-all, so grads flow.

**MLA gets the identical treatment** (this revision corrects the 07-21
draft, which wrongly assumed MLA would use `apply_cp_to_attention_module`
ring/zigzag -- it did not, and there is no evidence the custom
`inner_attention` SDPA path is compatible with torchtitan's ring dispatcher
without separate work). Apply the same seq-local-project -> all-to-all ->
head-subset-SDPA -> all-to-all-back -> o_proj pattern MLA already partially
has (it already head-shards the SDPA step; the missing piece is moving the
all-gather to an all-to-all and doing it BEFORE the projections instead of
after).

## Staged verification plan (2026-07-23 decision: 4 cards first)

Do NOT jump straight to a large joint mesh. Order:

1. **4 cards, single-axis real Ulysses CP (cp=4, no TP)**: replace the
   all-gather with the all-to-all wiring above for BOTH KDA and MLA. Parity
   gate: cp=4 vs cp=1 loss within bf16 band (mirrors the existing
   EIGHTCARD cp=2/cp=4 all-gather-era numbers, now on the real Ulysses
   path). Also re-verify CP+FSDP / CP+EP / CP+PP still compose (Problem 1's
   fix changes CP's internals; the external contract should be unchanged,
   but re-run the matrix, don't assume).
2. **4 cards, CP+TP joint (the actual new capability)**: smallest useful
   grid, e.g. `cp=2, tp=2`. This is where the head-divisibility check and
   the 2D-mesh sub-group derivation get proven or broken. Fail loudly if
   head counts don't divide cleanly rather than silently misrouting.
3. **Only after (1) and (2) are both green**: scale card count up for the
   fuller joint matrix (CP+TP+FSDP, CP+TP+EP, CP+TP+PP, and however much of
   the full 5D mesh the card count allows) -- same discipline as the
   existing LoRA-x-parallelism and full-param matrices in
   EIGHTCARD_VERIFICATION.

Getting the fail-loud fix (Problem 2) in immediately, ahead of the 4-card
run, means an accidental `tp+cp` config during step (1)'s development
cannot silently produce a misleading "it trained" result.

---

## Original 2026-07-21 draft (superseded above; kept for the numerics proof, which is still valid)

Turns the RFC's "CP out of scope" into a specced, numerically-validated
plan. The novel/risky numerics are DONE (bit-exact); what remains is
standard-but-nontrivial wiring.

### What is proven (kda_ulysses_cp_probe.py)

`chunk_kda` is **bit-exactly per-head independent** (full[:h] vs
head-subset rel-err 0.00 -- each head carries its own delta-rule state).
Therefore Ulysses head-sharding is numerically EXACT: seq-shard ->
all-to-all(seq<->head) -> chunk_kda on the head subset over the FULL
sequence -> all-to-all back reconstructs the non-CP reference with
**rel-err 0.00 at cp=2 and cp=4**.

### Verification plan (once wired)

- Layer parity: KDA layer under cp=2 vs cp=1, rel-err within bf16 band.
- Full-model parity: `torchtitan.train` cp=2 vs cp=1 loss (same-init via
  checkpoint), and CP composed with FSDP/PP/EP.
