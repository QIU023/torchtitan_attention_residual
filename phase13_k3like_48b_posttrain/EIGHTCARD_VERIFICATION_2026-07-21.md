# 8-card K3 support verification (2026-07-21)

Full multi-GPU smoke + accuracy sweep on the 8x5090 box, ahead of moving
the heavy 48B QLoRA/QAT SFT/GRPO to 2xH200. torchtitan fork @129e29de+.

## Full-param parallelism matrix (kimi_linear_debugmodel, 8 cards)

| cell | step-1 | step-3 loss / grad_norm |
|---|---|---|
| FSDP (dp8) | 7.613 | 6.797 / 0.66 |
| TP (dp4,tp2) | 7.657 | 7.022 / 2.31 |
| EP (dp8,ep2) | 7.614 | 6.741 / 0.62 |
| PP (dp4,pp2) | 7.656 | 6.885 / 1.25 |
| TP+EP (dp4,tp2,ep2) | 7.671 | 7.034 / 2.25 |
| TP+PP (dp2,tp2,pp2) | 7.663 | 7.170 / 4.17 |
| **4D FSDP+TP+EP+PP (dp2,tp2,pp2,ep2)** | 7.676 | 7.252 / 3.39 |

step-1 losses in a tight 7.61-7.68 band (same random init); step-3 diverge
within the bf16/KDA-nondeterminism band. `run_fullparam_matrix.sh`.

**TP was the untested axis** (the LoRA matrix had FSDP/EP/PP). TP+EP and the
full 4D mesh needed a **core MoE fix** (fork 129e29de): under TP+EP the
router outputs are DTensors sharded on the token dim, and the routing-map
one-hot `scatter_` had no DTensor placement strategy that preserves
Shard(dim=1) -- in-place errored, out-of-place replicated (breaking the
downstream Partial(sum) token-count contract). Fix: scatter on the local
shard + rewrap with the router's placement. This is core common-MoE (helps
any MoE under TP+EP), flagged for the PR as a core change. LoRA matrix
(bf16 + MXFP4) separately green -- see MXFP4_LORA_VERIFICATION.

## K3 feature smokes under parallelism

- **Gated MLA** (k3faithful flavor): FSDP+EP green. 4D fails on a
  provisional-gate mixed Tensor/DTensor mul under TP (the gate param isn't
  TP-sharded); follow-up, and the gate form reconciles at 7.27.
- **Deltas composed under FSDP2/8gpu** (`mgpu_capstone_deltas.py`): Gated
  MLA + alpha-graft + MXFP4/MXFP8-QAT (12 wraps) + Per-Head Muon trained
  together, loss finite. Notably **Muon's Newton-Schulz orthogonalization
  works on sharded DTensor params** under FSDP2.
- **Quantile Balancing**: the delta function is unit-tested; the optimizer
  pre-step hook integrates at the trainer level (needs optimizers /
  parallel_dims), not a standalone smoke.
- **EP@896 wide-EP** (`ep896_construction_smoke.py`): 896 experts, EP@8,
  112/rank, forward finite.

## Context Parallel -- LANDED + composes (was hard-blocked)

CP is no longer a NotImplementedError. Correctness-first integration
(torchtitan fork @f4b6f46f): both KDA and MLA layers all-gather the seq
shard at their boundary, run the unchanged forward on the full sequence,
slice this rank's shard from the output. Differentiable all_gather ->
reduce-scatter backward; FSDP's mesh includes cp (parallel_dims line 217)
so param grads reduce over cp.

- **Forward correct**: cp=2 vs cp=1 at the SAME dp (dp1) -> step-1 loss
  7.65969 vs 7.65907 (delta 6e-4, within bf16). (Different-dp comparisons
  diverge only via init-RNG, as with TP.)
- **Composes on 8 cards**: CP+FSDP (dp4,cp2) 6.751, **PP+CP+FSDP
  (dp2,cp2,pp2) 7.059**, CP+EP (dp4,cp2,ep2) 6.751, CP4+FSDP (dp2,cp4)
  6.891 -- all train.

Correctness-first = KDA/MLA compute replicated across cp ranks (no memory
saving yet); the memory-optimal path (KDA Ulysses head-shard -- bit-exact,
+ MLA SDPA ring) is the optimization, specced in CP_ULYSSES_DESIGN. CP+TP
out of scope (all-gather guards on plain non-DTensor activations). Note:
we have NO real 1M-context data -- CP here is infra/numerics verification
on short seqs (the capability CP enables), not a long-context training run.

## MXFP4 x {EP, PP} directly -- blocked on the meta-first order

torchtitan.train materializes as build(meta) -> parallelize(FULLY_SHARD)
-> to_empty -> init_weights, so init runs on ALREADY-SHARDED DTensor
params. MXFP4's validated path is quantize-THEN-shard (Phase 0); a
post-init quantize hook would instead quantize sharded DTensors and
register FSDP-unmanaged split-storage (base_qdata/base_scale) -- a real
architectural mismatch, not a hook. Direct MXFP4x{EP,PP} needs either a
DTensor-aware shard-then-quantize or a build-on-device parallelize path;
not landed. The composition is inferred (bf16-LoRA composes with EP/PP;
MXFP4 base shards under FSDP2) but not directly run through the trainer.

## KDA context parallel via Ulysses (RFC future-work item)

`kda_ulysses_cp_probe.py`: chunk_kda is **bit-exactly per-head independent**
(full[:h] vs head-subset rel-err 0.00), so Ulysses head-sharding is exact:
seq-shard -> all-to-all -> head-shard -> chunk_kda on full seq -> all-to-all
back reconstructs the non-CP reference at **rel-err 0.00 (cp=2 and cp=4)**.
The RFC flagged CP-for-KDA as unsupported (fla chunk_kda); the head-shard
numerics are now proven. Remaining: full-layer wiring (seq-local proj +
conv halo + torchtitan context_parallel context) -- standard engineering.

## Deterministic numerical parity (seed + --debug.deterministic)

KDA is deterministic under `--debug.seed 42 --debug.deterministic`
(bit-identical loss across repeat runs). With a FIXED global batch
(local_batch=1, global_batch=8) so data is identical across configs:

- **EP is BIT-EXACT to FSDP** (step-1 loss 7.59001 == 7.59001) -- EP is
  numerically transparent (experts sharded, same math).
- **TP / PP / 4D differ from FSDP at step-1 by ~0.06** -- NOT a numerics
  bug: sharded weight init consumes the seeded RNG differently (TP shards
  more params -> different initial weights), so absolute step-1 loss is
  not a valid cross-config metric. TP+EP == TP bit-exact (EP adds
  nothing). All configs train (loss descends, grad finite). A same-init
  cross-config parity needs a shared checkpoint load (below); the
  init-divergence is the standard torchtitan cross-parallelism caveat.

## DCP checkpoint save / load

- Save under FSDP(dp8): 8 `.distcp` shards written; **load: all 8 ranks
  load OK, no error** (same-mesh round-trip mechanically verified).
- Follow-up: resume-and-continue-training didn't cleanly advance on the
  debug flavor (tiny c4_test dataset likely exhausts) and a cross-degree
  reshard load (dp8 ckpt -> dp2) SIGTERM'd -- both need a non-debug
  fixture to verify the full-mesh reshard + resume. Save/load primitives
  work; the reshard+resume path is the open item.

## Open (documented, not silently skipped)

- TP-composition of the PROVISIONAL Gated-MLA gate (mixed-DTensor mul);
  reconciles 7.27.
- MXFP4 x {EP,PP} directly (no post-load quantize hook in train path).
- 48B QLoRA/QAT SFT + GRPO -> 2xH200.
- Ulysses full-layer CP integration into torchtitan's CP context.
