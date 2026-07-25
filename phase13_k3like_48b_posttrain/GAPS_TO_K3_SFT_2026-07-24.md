# Remaining gaps to K3-architecture QLoRA / full-param training+SFT (2026-07-24)

> **STATUS (end of session, same day): A1-A5, B6, B7 CLOSED** on this
> box -- see CP_TP_3D_VERIFICATION_2026-07-24.md Parts 2-3 and
> SESSION_HANDOFF_2026-07-24.md. B9's "cosmetic" grad_norm symptom
> turned out to be a REAL bug (FSDP skipped at dp1+cp -> unsynced cp
> replicas), fixed in `a42be25f`. Still open: B8 (7.27-gated), C10-C12
> (bigger hardware), TP x packed-MXFP4 base, upstream PR extractions.

Audit requested after the CP/Ulysses fix: what ELSE stands between the
current stack and reliable K3-arch QLoRA or full-param SFT. Ordered by
(blocks-training > silently-wrong > perf/cosmetic), each with where it
bites and the concrete fix. Context: CP x TP x PP 3D is now green
([CP_TP_3D_VERIFICATION_2026-07-24.md](CP_TP_3D_VERIFICATION_2026-07-24.md)).

## A. Correctness / blocks-training

1. **QLoRA (MXFP4/NF4-base) cannot go through `torchtitan.train` yet** --
   the known meta-first vs quantize-then-shard conflict (SESSION_HANDOFF
   section 5). titan builds meta -> shards -> inits; a quantized base
   needs real weights BEFORE sharding (quantize-then-shard) or a
   streaming per-tensor quantize-on-load. The verified MXFP4-LoRA path
   (`mxfp4_lora_fsdp_real.py`) builds on-device first -- fine at debug
   scale, needs ~96GB on one GPU at 48B. Fix: state_dict_adapter-level
   streaming quantize (load HF shard -> quantize -> distribute to the
   FSDP shard owner, never materializing the full model), or DCP-side
   pre-quantized checkpoint + a load hook. This is THE infra item for
   48B QLoRA on small-VRAM fleets.

2. **LoRA flavor without FSDP (dp_shard=1) crashes** fp32-vs-bf16 at the
   `block_attn_res` einsum (found today; PREEXISTING -- repros on
   7d8acabe with plain dp1+tp2, unrelated to CP). Root cause: without
   FSDP mixed-precision, the LoRA flavor leaves AttnRes projections fp32
   while the stream is bf16. Bites any single-axis debug run of the LoRA
   flavor (tp-only, cp-only, pp-only) -- exactly the runs used to bisect
   parallelism bugs. Fix: make the flavor's dtype story explicit (cast
   AttnRes proj/norm params to the training dtype at build, or run the
   einsum in the stream dtype); one-file change + a dp1 smoke in tests.

3. **AC x CP untested** -- every cell today ran activation_checkpoint
   off (debugmodel default). Under `full` AC the checkpointed layer
   recomputes the CP all-to-alls in backward; collectives inside
   recompute regions are legal only if all cp ranks recompute in
   lockstep (they should -- same schedule -- but this is exactly the
   kind of "should" the 07-23 doc warns about). 48B SFT WILL need AC.
   Action: one AC=full + cp2 (+tp2) cell before any real long run.

4. **DCP save/load under CP+TP untested** -- all today's cells ran
   checkpoint off. EIGHTCARD verified DCP under FSDP(dp8) incl.
   resharding to 4 ranks; the (fsdp x tp x cp) mesh save + the
   LoRA-adapter-only save under that mesh are unverified. Action: save +
   resume smoke on fsdp2tp2cp2, and a cross-mesh resume (save cp2 ->
   load cp1) since SFT recipes reshard at deploy.

5. **verl engine does not plumb CP** (and must inherit the no-balancer
   contract). verl's torchtitan engine does its own input prep -- if CP
   is exposed there without the contiguous-shard guarantee, the headtail
   bug class returns through the verl door. Action: when adding
   `context_parallel_degree` to the verl engine config, shard inputs
   contiguously (or reuse titan's prepare fn with balancer None) and
   assert `seq_len % cp == 0`; add the same parallelize-time ValueError.

## B. Verification debt (works-probably, unproven)

6. **Full 5D needs >= 16 ranks AND a bigger-H debug flavor** --
   debugmodel H=4 binds at tp*cp=4 (tp2cp4/tp4cp2 impossible). Cheap
   enabler either way: an H=8 debug flavor (d=512 or head_dim=32).
   Largest 8-rank cells (tp2cp2pp2, fsdp2tp2cp2, tp2cp2ep2) are done.

7. **Per-Head Muon x TP unverified** -- Muon's Newton-Schulz on sharded
   DTensors is verified under FSDP2 only (EIGHTCARD). Full-param K3
   recipe wants Muon under FSDP x TP (x CP). One capstone cell.

8. **Gated-MLA under the full mesh** -- today's fix (attn_gate_proj
   ColwiseParallel) un-crashed gated x TP and gated-LoRA x FSDPxTPxCP,
   but the gate FORM is provisional until 7.27 weights; re-run the
   deltas-compose capstone on the new mesh after reconciliation.

9. **grad_norm metric under CP prints the local partial norm** (clipping
   is globally consistent -- verified by rank-identical losses; sqrt-sum
   of cp4 partials reconstructs cp1's norm). Cosmetic but it WILL send
   someone chasing a phantom divergence. One-line metrics fix.

   > **WRONG -- corrected 2026-07-25**
   > ([PACKED_TP_VERIFICATION_2026-07-25.md](PACKED_TP_VERIFICATION_2026-07-25.md)
   > sec 4). The printed value IS the true global norm, and the APPLIED
   > gradient is under-scaled by the fsdp mesh size (`dp_shard x cp`) --
   > exactly cp, not sqrt(cp), and dp_shard is affected too. Cause:
   > kimi_k3's private `apply_fsdp` omits the
   > `disable_fsdp_gradient_division` call that the shared `apply_fsdp`
   > makes, while the loss is already `local_sum / global_valid_tokens`.
   > Measured with `cp_grad_scale_probe.py`, not inferred from curves
   > (AdamW is scale-invariant, so no loss curve can see it). Clipping is
   > therefore NOT globally consistent: it engages `dp_shard*cp` times
   > later than configured. Not a metrics fix; fix identified, not landed.

## C. Scale / perf (not correctness)

10. **1M-context**: memory contract now Ulysses-correct, but real
    long-ctx needs (a) a long-context dataset/loader path, (b) big-VRAM
    cards, (c) probably CP x PP together (both are now available). The
    16GB box demonstrated 32k on debugmodel.
11. **torchao 0.17 has no weight-only MXFP4 linear** -> QLoRA base
    dequant-matmuls (slow, correct). Re-check at torchao bumps; swap to
    the native kernel when it lands.
12. **Full-param 48B SFT stays H200-class** (fp32 masters ~576GB); this
    box changes nothing there. QLoRA is the realistic 48B path on
    consumer VRAM -- which is why A1 is the top item.

## Suggested order (if the goal is "K3 QLoRA SFT works end-to-end")

A2 (small, unblocks debug bisects) -> A3 + A4 smokes (one session) ->
A1 streaming-quantize (the real build) -> A5 verl CP plumbing ->
B6 flavor + 5D when >= 16 ranks are rented -> B7/B8 capstones ->
7.27 reconciliation gates everything provisional.
