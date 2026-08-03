# CP x TP x PP 3D fix: design + execution plan (2026-07-24, 8x5060Ti box)

Goal: **CP, PP, TP compose correctly** (debug model, 8x RTX 5060 Ti 16GB),
per CP_ULYSSES_DESIGN.md's staged plan. This doc is the concrete engineering
design for that work plus the findings from today's code re-read that the
07-23 doc did not have.

## 0. New findings from today's code re-read (beyond CP_ULYSSES_DESIGN)

### 0a. The TP+CP failure is WORSE than "CP silently no-ops": it is split-brain

CP_ULYSSES_DESIGN says the CP block is "silently skipped at every forward
call" under TP. Re-reading model.py shows the two attention types diverge:

- **MLA** (model.py L352): guard is
  `cp_group is not None and not isinstance(x, DTensor)` -> under TP x IS a
  DTensor (input_layernorm is `NoParallel()` with no use_local_output, so
  its output stays DTensor) -> **MLA skips CP**.
- **KDA** (model.py L601): forward does `_to_local_if_dtensor(x)` FIRST,
  and its CP guard has **no DTensor check** -> **KDA still runs CP** under
  TP.

Net effect of `tp>1, cp>1` today: input arrives seq-sharded [B, T/cp, D];
KDA layers all-gather and see the true full sequence; MLA layers treat the
local shard as the whole sequence -> **block-diagonal causal attention in
every MLA layer** while KDA layers are correct. The model trains, loss is
plausible, numerics are silently wrong. This is not "you get TP only".

### 0b. Suspected real bug in landed CP: headtail load balancer permutes the sequence

`trainer.py:669` calls `prepare_context_parallel_input(...,
config.parallelism.context_parallel_load_balancer)` and the default
(`configs.py:211`) is `"headtail"`. Under headtail, rank r's shard is
[chunk_r, chunk_{2cp-1-r}] of a 2cp-chunk split -- NOT a contiguous slice.
The kimi CP path reassembles with `torch.cat(all_gather(x), dim=1)` which
assumes contiguous rank-ordered chunks, so the "full sequence" the
KDA scan / MLA SDPA sees is **permuted** ([r0_head, r0_tail, r1_head,
r1_tail] instead of natural order). Causal semantics are broken.

Why the EIGHTCARD parity (6e-4, step-1) did not catch it: step-1 loss on a
random-init model is ~ln(vocab) regardless of token order -- a permutation
of the input barely moves it. The parity gate was too weak for this bug
class. **Action: empirically confirm with a multi-step deterministic run
(cp2-headtail vs cp2-nolb vs cp1), then force `load_balancer=None` for
kimi_k3 with loud validation.** Note load balancing is pointless under
Ulysses anyway: every rank computes all T for its head subset, so per-rank
work is symmetric by construction; headtail exists for ring-CP where ranks
own causal seq chunks.

### 0c. Gated MLA gate is TP-broken (known from EIGHTCARD "4D fails" note)

`attn_gate_proj` is absent from the TP plan -> plain-tensor param meets
DTensor x -> mixed mul crash in the k3faithful flavor under TP. Fix while
we are in the file: `ColwiseParallel(use_local_output=True)` shards the
per-head gate across TP (out_features = num_heads), and the gate slice
must then follow the same head-subset bookkeeping as q/k/v under CP.
(The debugmodel default flavor has mla_gated off, so the 3D matrix gates
on the base flavor first; gated flavor is a follow-up cell.)

## 1. Design: real Ulysses CP, structured for TP composition from day one

CP_ULYSSES_DESIGN's staged plan says "Ulysses single-axis first, then
CP+TP". We keep that order for VERIFICATION, but the implementation is
written once, TP-aware from the start, because the correct code structure
is the same in both cases:

**Principle: all CP communication happens in plain-tensor land, in the
gap where the current code already strips DTensor.** Under TP, projections
produce head-sharded DTensors (MLA) or replicated locals (KDA); we
`to_local` right after the projections, do the Ulysses all-to-all on the
CP sub-mesh over plain tensors, and re-wrap only where the TP plan needs a
DTensor again (MLA o_proj's Rowwise input). This is option (b) from
CP_ULYSSES_DESIGN, chosen because it mirrors the `_to_local_if_dtensor`
pattern the file already uses everywhere and never asks DTensor to
dispatch a multi-axis collective.

### 1a. Mesh + groups

- `parallel_dims.get_mesh("cp").get_group()` already yields the correct
  CP sub-group per (dp, tp, pp) coordinate -- the existing wiring is
  reused unchanged. TP's own collectives ride the DTensor machinery on
  the tp mesh; no cross-axis group is ever needed.
- Head divisibility (validated loudly at parallelize time, ValueError):
  - MLA: `num_attention_heads % (tp * cp) == 0`
  - KDA: `kda_num_heads % cp == 0` (KDA is NoParallel under TP -- no tp
    factor; its redundancy across TP ranks is unchanged)
  - seq: `seq_len % (2*cp) == 0` if a balancer were on; with balancer
    forced None, `seq_len % cp == 0` (trainer shard) -- validated too.

### 1b. KDA forward under Ulysses CP (both with and without TP)

x arrives [B, T/cp, D] (plain, or DTensor(Replicate) -> to_local, same as
today). All projections run seq-local at T/cp:

```
q/k/v = q/k/v_proj(x)              # [B, T/cp, H*K]  seq-local
g_raw = f_b(f_a(x)); g_out = g_b(g_a(x)); beta = b_proj(x)
-> view heads: [B, T/cp, H, *]
-> all_to_all(seq<->head, cp) on q,k,v,g_raw,g_out,beta
   -> [B, T, H/cp, *]              # full seq, head subset
-> conv1d on q/k/v with weight channels sliced to the head subset
   (depthwise conv: channel slice [h0*K : h1*K] is exact)
-> fused_kda_gate(g_raw, A_log[h0:h1], dt_bias[h0*K:h1*K])
-> chunk_kda on the head subset over full T   (bit-exact per-head:
   kda_ulysses_cp_probe rel-err 0.00)
-> o_norm(o, g_out)                # head-local
-> all_to_all back -> [B, T/cp, H, K] -> o_proj -> [B, T/cp, D]
```

No rank ever holds [B, T_full, D] or full-head activations: memory is
O(T/cp * H) + O(T * H/cp) per tensor. This is the actual Ulysses memory
contract (Problem 1 fixed).

Conv correctness note: the conv is causal within the full gathered
sequence, so no halo is needed (the a2a delivered full T for this head
subset). This matches the design doc's step 3.

### 1c. MLA forward under Ulysses CP (TP-aware)

x arrives [B, T/cp, D] plain (no TP) or DTensor(Replicate on tp) (TP).

```
q = q_proj(x)                      # Colwise under TP -> DTensor Shard(-1)
compressed_kv = kv_a_proj_with_mqa(x)   # NoParallel -> Replicate
kv = kv_b_proj(kv_a_layernorm(k_pass))  # Colwise -> Shard(-1)
-> to_local ALL of them            # [B, T/cp, (H/tp)*d] locals
-> view heads: q [B, T/cp, H_loc, qd], kv [B, T/cp, H_loc, nd+vd]
-> all_to_all(seq<->head, cp): q,kv -> [B, T, H_loc/cp, *]
-> k_rot (headless, [B, T/cp, rope]): all_gather over cp -> [B, T, rope]
   (broadcast to the head subset; tiny tensor, gather not a2a)
-> SDPA on [B, H_loc/cp, T, *] (causal, full seq)
-> all_to_all back -> [B, T/cp, H_loc, vd] -> flatten heads
-> (TP) DTensor.from_local(.., tp_mesh, [Shard(-1)]) -> o_proj Rowwise
   (all-reduce over tp) | (no TP) plain o_proj
-> [B, T/cp, D]
```

The `inner_attention` PrepareModuleInput(use_local_output=True) TP hook
already strips DTensor before SDPA; with the explicit to_local this
becomes a no-op pass-through -- kept for plan compatibility.

Gated MLA: gate = sigmoid(attn_gate_proj(x)) is per-head [B, T/cp, H].
Under TP it becomes Colwise -> [B, T/cp, H/tp] local; under CP the gate
must be a2a'd with the heads (it multiplies attn_out BEFORE heads are
re-gathered in the new layout... actually AFTER a2a-back it is seq-local
again, so apply the gate after the a2a-back at [B, T/cp, H_loc, vd] --
gate needs no a2a at all). Deferred to a follow-up cell after base 3D.

### 1d. Autograd

`torch.distributed.nn.functional.all_to_all_single` is differentiable
(backward = transposed a2a), same contract as the probe's
`all_to_all_headseq`. The probe validated round-trip rel-err 0.00 at
cp=2/4. We lift `all_to_all_headseq` from
`phase13_k3like_48b_posttrain/kda_ulysses_cp_probe.py` into the model as
a shared helper.

### 1e. What is NOT changing

- Module boundary contract: seq-sharded plain tensor in / out. PP
  send/recv, AttnRes block stacking, EP token dispatch, FSDP wrapping all
  see exactly what they saw before -> the proven FSDP/EP/PP composition
  carries over structurally (re-verified, not assumed).
- FSDP mesh already includes cp in dp_shard_cp (grads reduce over cp) --
  unchanged.
- The old all-gather SP path is REPLACED, not kept as a flag. One code
  path; fewer silent divergences.

## 2. Execution order (each step gated on the previous)

0. **Baseline + bug confirmation** (before touching code):
   a. 1-GPU debugmodel smoke (env sanity on sm_120).
   b. cp1 vs cp2-headtail vs cp2-nolb, seed=42 deterministic, 20 steps:
      confirm/refute 0b. Also reproduces the 6e-4-band step-1 parity.
1. **Loud guards** (small commit, independent):
   - ValueError at parallelize time if `cp_enabled and tp_enabled` --
     REPLACED at step 3 by real support; interim protection.
   - Force/validate `context_parallel_load_balancer is None` for kimi_k3
     (ValueError with the honest explanation), if 0b confirms.
   - Head/seq divisibility ValueErrors (1a).
2. **Ulysses rewrite, single-axis verification** (the big commit):
   - KDA + MLA per 1b/1c. Remove the DTensor guard asymmetry (0a).
   - Gates: cp2 & cp4 vs cp1 multi-step deterministic parity (not just
     step-1); CP+FSDP, CP+PP, CP+EP cells re-run; peak-mem check
     cp4 < cp1-per-rank at fixed global seq (the memory contract).
3. **CP+TP on** (remove the step-1 guard):
   - to_local/a2a/from_local wiring live under TP; divisibility tp*cp.
   - Gates: tp2cp2 (dp1) vs cp1tp1 parity band; tp2cp2 vs tp2cp1.
4. **3D + composition matrix on 8 cards** (the deliverable):
   - tp2 x cp2 x pp2 (dp1) -- THE 3D target cell
   - fsdp2 x tp2 x cp2; tp2 x cp2 x ep2; tp2 x cp4; tp4 x cp2 (head
     limits permitting); LoRA x (tp2 x cp2) smoke
   - regression cells: fsdp8, tp2, pp2, cp2 singles unchanged vs step-0
     baselines.
5. **Docs**: update CP_ULYSSES_DESIGN.md status + EIGHTCARD-style
   verification report; RFC bullet-1 caveat can then be upgraded.

## 3. Debug model head-count check (before running anything)

kimi_k3_debugmodel: verify `num_attention_heads` and `kda_num_heads`
divide tp*cp in {2x2, 2x4, 4x2} -- read from model_configs.py at step 0;
if not, add a 3D-friendly debug flavor rather than bending the checks.

## 4. Out of scope today (kept honest)

- 1M-context run: Ulysses fixes the memory scaling, but this box (16GB)
  verifies the mechanism at debug seq lens, not the 1M target.
- Gated-MLA TP sharding beyond the base cell (0c) -- follow-up.
- verl-engine CP plumbing (verl side does its own input prep; K3 SFT
  via verl with CP needs the same shard+balancer contract checked there).
- fla-core conv halo seq-sharding (obsoleted by Ulysses: conv runs on
  full-T head-subset, no halo needed).
