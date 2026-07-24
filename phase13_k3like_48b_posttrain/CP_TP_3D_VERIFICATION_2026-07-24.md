# CP x TP x PP 3D verification report (2026-07-24, 8x RTX 5060 Ti 16GB)

Companion to [CP_TP_3D_FIX_DESIGN_2026-07-24.md](CP_TP_3D_FIX_DESIGN_2026-07-24.md)
(design + the two new findings). Reproduce with
[run_cp_tp_3d_matrix.sh](run_cp_tp_3d_matrix.sh).

Torchtitan fork commits (branch `attention_residual_dev`, on top of `7d8acabe`):

| commit | content |
|---|---|
| `331e52a7` | fail-loud guards: CP+TP NotImplementedError (interim), load-balanced CP ValueError, kimi flavors default `context_parallel_load_balancer=None` |
| `5b654384` | real Ulysses CP for KDA+MLA (fused a2a seq<->head, seq-local projections, channel-sliced conv), head-divisibility ValueErrors |
| `0f37ec4a` | CP+TP enabled (guard lifted); `attn_gate_proj` TP-wrapped (fixes standing gated-MLA-under-TP crash) |

All runs: `kimi_linear_debugmodel` (4 layers: 3 KDA + 1 MLA, H=4, d=256,
Block AttnRes, MoE 8 experts), `--debug.seed 42 --debug.deterministic`,
bf16, seq 512 unless noted.

## 1. Bug confirmations (why the guards exist)

**Headtail load balancer permuted the sequence under CP (real bug in the
landed 07-21..23 CP).** `trainer.py` shards CP inputs through
`prepare_context_parallel_input` whose default balancer is `headtail`
(rank r gets chunks `[r, 2cp-1-r]` of a 2cp-split). The kimi CP path
reassembled with `torch.cat(all_gather(...))` assuming contiguous
rank-ordered shards, so the "full sequence" seen by chunk_kda/SDPA was
`[c0, c3, c1, c2]` at cp=2: future-token leakage (c1 attends c3) plus
missing context (c3 cannot see c1/c2). Empirical (20-step deterministic):

| run | step-1 loss | delta vs cp1 |
|---|---|---|
| cp1 | 7.63564 | — |
| cp2, balancer None | 7.63565 | 1e-5 |
| cp2, headtail (old default) | 7.63212 | **3.5e-3 (350x)** |

Step-1-only parity on a random-init model (the EIGHTCARD gate) cannot
catch this bug class -- a random model is near-insensitive to input
permutation; the loss curve still descends plausibly. Fixed by defaulting
the balancer to None in all kimi flavors + a ValueError on any non-None
value. Balancing is unnecessary for Ulysses CP (per-rank work is
head-symmetric, not seq-causal).

**CP+TP was split-brain, worse than documented.** CP_ULYSSES_DESIGN said
CP "silently no-ops" under TP. Actually only MLA skipped CP (its guard
tested `not isinstance(x, DTensor)`); KDA strips DTensor before its guard
and still applied CP -> MLA ran block-diagonal attention on the local
shard while KDA saw the true full sequence. Interim NotImplementedError
(331e52a7), then real support (0f37ec4a).

## 2. What the Ulysses rewrite changes

Old (07-21): all-gather x to [B, T, D] BEFORE projections -> full-seq
activations on every rank at any cp (Megatron-SP structure, no memory
scaling), head-shard only on chunk_kda/SDPA. New (5b654384), both KDA and
MLA: projections seq-local at T/cp -> ONE fused differentiable all-to-all
(q/k/v/gates/beta concatenated on the feature axis) to full-seq
head-subset layout -> causal short conv (weights channel-sliced per head
subset; depthwise, verified bit-exact vs ShortConvolution) ->
fused_kda_gate (A_log/dt_bias sliced) -> chunk_kda / SDPA on H/cp heads
over full T -> a2a back -> o_proj at T/cp. Module boundary (seq-sharded
plain tensor in/out) unchanged -> FSDP/EP/PP composition preserved by
construction and re-verified below.

**Peak-memory A/B at cp4 (old all-gather CP needs
`--parallelism.context_parallel_load_balancer None` to be comparable):**

| seq_len | old (7d8acabe) | new Ulysses | delta |
|---|---|---|---|
| 8192 | 1.65 GiB | 1.34 GiB | -19% |
| 32768 (bs1) | 2.98 GiB | 2.33 GiB | -22% |

The relative gap grows with T and with model width (the eliminated term
is O(T x D) hidden-state + projection activations at full T; debugmodel's
d=256 understates the 48B case).

## 3. Parity (20-step deterministic, dp1)

| run | step-1 | step-20 | note |
|---|---|---|---|
| cp1 | 7.63564 | 3.81458 | baseline |
| cp2 (Ulysses) | 7.63565 | 3.86308 | step-1 delta 1e-5; 20-step drift 0.05, same band as the old verified all-gather CP (3.85553), within 0.007 of it every step |
| cp4 (Ulysses) | 7.63564 | — | step-1 EXACTLY equals cp1 |
| tp2 | 7.64027 | 3.95069 | TP's own known band vs cp1 (4.6e-3 step-1) |
| tp2cp2 | **7.64027** | 4.00261 | step-1 EXACTLY equals tp2-only -> CP adds <1e-5 on top of TP; 20-step drift vs tp2 is 0.05 = the cp2-vs-cp1 band |

Attribution is the point: the CP axis contributes the same (tiny) numeric
band whether or not TP is on.

## 4. Composition matrix (all PASS = descending finite loss)

| cell | ranks | result |
|---|---|---|
| **tp2 x cp2 x pp2 (dp1, 1F1B) -- the 3D target** | 8 | PASS 7.647 -> 5.60@8 |
| fsdp2 x tp2 x cp2 | 8 | PASS 7.649 -> 5.27@10 |
| tp2 x cp2 x ep2 (dp2) | 8 | PASS, losses bit-identical to fsdp2tp2cp2 (EP==FSDP parity holds under CP+TP) |
| cp2 x fsdp2 | 4 | PASS 7.588 -> 5.96@5 |
| cp2 x ep2 (dp4) | 8 | PASS 7.571 -> 5.83@5 |
| cp2 x pp2 x fsdp2 | 8 | PASS 7.653 -> 6.01@5 |
| LoRA x fsdp2 x tp2 x cp2 (`gated_lora` flavor) | 8 | PASS (trains; slow start is the zero-init adapter, expected) |
| gated k3faithful x tp2 | 2 | PASS -- **was a mixed Tensor/DTensor crash before 0f37ec4a** (standing EIGHTCARD failure, now fixed) |
| regression: fsdp8 / pp2fsdp4 / 4D fsdp+tp+pp+ep | 8 | PASS (non-CP paths untouched) |
| unit tests `experiments/kimi_k3/tests` | — | 79 passed + 64 subtests |

## 5. Known limitations / follow-ups

- **LoRA flavor without FSDP (dp_shard=1) crashes** with fp32-vs-bf16 at
  `block_attn_res` einsum -- PREEXISTING (repros at plain dp1+tp2 on
  7d8acabe, before any CP change): without FSDP mixed-precision the LoRA
  flavor's dtype story is inconsistent. Not CP-related; fix separately.
- Full 5D (FSDP x TP x CP x EP x PP all >1) needs >= 16 ranks + more
  heads; debugmodel H=4 binds head-divisibility at tp*cp=4. A 3D-friendly
  bigger-H debug flavor is the cheap enabler (tp2cp4 / tp4cp2 need H=8).
- grad_norm logged per-rank under CP is the LOCAL partial norm (metric
  display only -- clipping is globally consistent, verified by rank-
  identical losses at every step; sqrt-sum of the cp4 partials
  reconstructs cp1's 3.08). Preexisting behavior, worth a cosmetic fix.
- 1M-context: the memory CONTRACT is now right, but this box (16 GB)
  verifies mechanism at 32k on the debug model, not the 1M target.
- verl engine SFT-with-CP: verl does its own input prep; the
  contiguous-shard + no-balancer contract must be enforced there when CP
  is exposed through the verl backend (not yet wired).
