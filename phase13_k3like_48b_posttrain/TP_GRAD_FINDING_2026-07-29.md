# TP gradients are systematically attenuated (2026-07-29)

## What was measured

Inside a REAL trainer run -- KDA present, FSDP active, going through the
trainer's own `parallel_dims` -- with `clip_grad_norm_` patched to report the
MATERIALIZED global norm (every gradient's `full_tensor()`, which reduces Partial
and gathers Shard) next to the number the trainer prints. All arms load the SAME
seed checkpoint and see the same data; only the parallelism differs.

    dp2      reported 8.495697   materialized 8.495697   ratio 1.000000
    dp4      reported 8.756243   materialized 8.756243   ratio 1.000000
    dp2xtp2  reported 2.742631   materialized 2.745368   ratio 1.000998

**The reporting is correct.** `clip_grad_norm_` equals the materialized norm in
every configuration, TP included. The earlier "metric artifact" hypothesis is
dead, and so is any hope that this is only a logging problem.

Holding the DP degree fixed at 2 and adding only TP: **8.496 -> 2.743 and
8.703 -> 2.806, a consistent 3.10x**.

## Where it lives

Per-parameter materialized gradient norms, same seed, same step, bucketed by
module (dp2 vs dp2xtp2):

| module | n | dp2 | dp2xtp2 | ratio |
| --- | --- | --- | --- | --- |
| lm_head | 1 | 0.4358 | 0.4339 | **1.004** |
| mlp_res | 42 | 0.7737 | 0.4086 | 1.893 |
| shared_experts | 60 | 0.6851 | 0.3327 | 2.059 |
| inner_experts (routed) | 60 | 2.7338 | 1.3023 | 2.099 |
| latent MoE proj | 60 | 2.7246 | 1.0542 | 2.584 |
| self_attn q/k/v/o_proj | 66 | ~2.5 | ~0.69 | **3.59-3.73** |
| dense ffn gate/up/down | 3 | ~1.76 | ~0.47 | **3.72-3.74** |
| embed_tokens | 1 | 4.4929 | 1.2237 | **3.672** |

The pattern is not a single uniform factor, and it is not noise:

* `lm_head`, the parameter closest to the loss, is **unaffected** (1.004);
* the MoE family sits around 2;
* everything else, including the embedding at the far end of the backward pass,
  sits at 3.6-3.7;
* and TP is smaller in every single bucket -- a random divergence would scatter
  the ratios around 1, not put all of them on one side.

Monotone attenuation with distance from the loss is the signature of a gradient
losing a contribution as it propagates back through the TP-sharded path, not of
two runs that merely diverged.

## What this does and does not overturn

It does NOT show the earlier MoE-routing finding was wrong: routing divergence is
real and independently measured (magnitudes matching, directions differing, worst
at `router.gate`). But routing divergence cannot produce a one-sided 3.1x, so it
is at most a second effect layered on this one.

It DOES mean every TP result in this logbook is provisional. Loss curves did not
catch it because AdamW is scale-invariant -- a uniformly attenuated gradient
trains to a plausible-looking curve, which is exactly why the seed-checkpoint
comparison was worth building.

## Next instrument

The buckets point at the boundary between TP-sharded and replicated modules.
`lm_head` being clean and `embed_tokens` being 3.7x off, with the attention
projections in between, suggests checking the backward gradient placements at the
points where the TP plan converts between sharded and replicated -- the same
class of bug as the `block_attn_res` Partial-vs-Replicate grad placement fixed
earlier, which also produced a wrong-magnitude gradient that no loss curve
revealed.

## Decomposition (same day, after the depth profile)

The ratio compounds with depth rather than being one global factor -- with KDA,
layer 20 (nearest the loss) is 1.066 and layer 0 is 3.613, about 1.063x lost per
layer. So the next question was which module carries it. Two diagnostic flavors
hold everything else fixed:

| configuration | dp2 | dp2xtp2 | ratio | layer-0 ratio |
| --- | --- | --- | --- | --- |
| k3mini (KDA + MoE) | 8.496 | 2.743 | **3.10** | 3.613 |
| no KDA (MoE kept) | 8.933 | 5.020 | **1.78** | 1.865 |
| dense MLA only | 18.138 | 16.227 | **1.12** | -- |

Three contributors, all compounding with depth, KDA the largest.

## Why this is probably not a gradient-placement bug -- and why the instrument
## cannot settle it

Dense-MLA-only is pure tensor sharding of dense matmuls, which is exactly
equivalent mathematically, and it still shows 1.12. That residual is the tell:
TP changes the reduction order of every matmul, which in bf16 perturbs
activations at ~1e-3, and this model AMPLIFIES perturbations by roughly 1.6x per
layer (measured independently in the CP parity probe). Over 21 layers that is a
factor of ~2e4, so any perturbation saturates long before the output.

Once a perturbation saturates, a whole-model gradient-norm ratio cannot separate
"a term is missing from the backward" from "two runs diverged and amplified".
The three numbers above are then explained without any bug: TP perturbs
activations (bf16), MoE turns a perturbation into a discrete routing flip, and
KDA's recurrence carries state along the sequence so it amplifies hardest. That
ordering matches the measurements.

This is the same trap as the CP logits comparison earlier in this logbook, where
a correct path and a deliberately broken control both landed at ~0.7. The
saturation lesson has now cost two investigations, so it is worth stating as a
rule: in this model, any end-to-end numerical comparison across parallelism is
uninformative beyond the first few layers.

What would settle it: a SINGLE-layer TP comparison, or a first-layer-only
gradient check, where amplification has not yet run. Until that is done, the
honest status is "unexplained, most likely amplified divergence rather than a
lost gradient term" -- not "TP is broken", and not "TP is fine".

The one thing measured cleanly and repeatedly: `clip_grad_norm_` reports exactly
the materialized norm in every configuration, so nothing here is a reporting
defect.

## Single-layer measurement: it IS a defect, and it is in the MLA path

Whole-model ratios were uninformative because perturbations saturate. One layer
removes amplification entirely, and the picture inverts:

| 1 layer | dp2 | dp2xtp2 | ratio |
| --- | --- | --- | --- |
| dense MLA | 10.339 | 9.709 | **1.0649** |
| dense MLA, AttnRes off | 9.837 | 9.398 | **1.0468** |
| MLA + MoE | 8.865 | 7.474 | 1.1862 |
| KDA | 2.519 | 2.579 | **0.9766** |
| **upstream llama3_debugmodel** | 1.529 | 1.542 | **0.9915** |

KDA is CLEAN (0.977). It only looked like the largest contributor in the
multi-layer runs because its recurrence amplifies whatever it is fed -- the
opposite of the conclusion the depth profile suggested.

The upstream llama3 control is what makes this actionable: under the identical
measurement, in the same trainer, with the same TP degree and batch, a known-good
model is within 0.9%. So TP itself is numerically sound here, and a ~4.7% gap in
a single dense MLA layer is ours.

AttnRes accounts for a small part of it (1.065 -> 1.047 with it disabled) but not
the bulk, so its hand-rolled Partial-on-tp grad placement is not the main
culprit.

And 1.047^21 = 2.7, 1.065^21 = 3.7 -- which brackets the 3.10x whole-model ratio
and matches the 3.6-3.7 seen on the attention projections. The per-layer defect
fully explains the end-to-end number.

Remaining suspects, in the MLA TP plan: the inner_attention
PrepareModuleInput(use_local_output=True) boundary, and the Gated-MLA output gate
registered as ColwiseParallel(use_local_output=True) -- both convert between
sharded and replicated, which is the class of boundary that produced the earlier
block_attn_res placement bug.

## FOUND: the Gated-MLA output gate

| 1-layer dense MLA | dp2 | dp2xtp2 | ratio |
| --- | --- | --- | --- |
| full (gate + AttnRes) | 10.339 | 9.709 | 1.0649 |
| AttnRes disabled | 9.837 | 9.398 | 1.0468 |
| **gate disabled** | 12.478 | 12.731 | **0.9801** |
| upstream llama3 (control) | 1.529 | 1.542 | 0.9915 |
| KDA (control) | 2.519 | 2.579 | 0.9766 |

Disabling `attn_gate_proj` moves the ratio from 1.047 to 0.980 -- into the same
band as both controls. The gate is the primary carrier.

Why it is the gate, mechanically. `attn_gate_proj` is registered
`ColwiseParallel(use_local_output=True)` and its INPUT is `x`, the replicated
residual stream, which the module also feeds to the attention path. A colwise
matmul on a replicated input produces a gradient w.r.t. that input which is
PARTIAL -- each rank holds one term of a sum that must be all-reduced over the tp
axis before it can be added to the residual's other gradient contribution. If
that reduction is skipped, the residual keeps only 1/tp of the gate's
contribution, which is a systematic under-count in exactly one direction, per
layer, compounding with depth. Every symptom matches.

This is the same defect shape as the `block_attn_res` fix recorded earlier in
this logbook: "a bare to_local() defaults the backward grad placement to the
DTensor's own placement (Replicate), which tells DTensor the gradient is already
consistent and SKIPS the all-reduce". The gate was added later and never got the
equivalent treatment.

Note the ratio lands slightly BELOW 1 without the gate (0.980), as do both
controls (0.9915, 0.9766). That residual is bf16 reduction-order noise and is not
one-directional across configurations, unlike the gate's contribution.

### Fix direction (not yet implemented)

The gate must either keep its output in DTensor space so the chain stays
consistent (as `q_proj` / `kv_b_proj` do with `use_local_output=False`), or
convert with explicit grad placements that force the tp-axis all-reduce, the way
`block_attn_res` does. The first is cleaner but changes the dtype/kind of
`attn_out * gate`, which is why `use_local_output=True` was chosen originally --
so the fix needs the multiply and `o_proj`'s input handling adjusted together,
and re-verified on this same single-layer measurement.

## RETRACTION: the gate hypothesis is disproven, and the localization was unsound

Two things are now settled, both against me.

**The mechanism was wrong.** The proposed defect was that `attn_gate_proj` reads a
REPLICATED plain `x` with a column-sharded weight, leaving `dL/dx` Partial and
unreduced. A runtime print says `x` arrives at the gate as a **DTensor**, not a
plain tensor:

    [TPDIAG] gate input: group=set is_dtensor=True type=DTensor

So DTensor's autograd redistributes that gradient itself and there is nothing to
reduce by hand. The explicit all-reduce added for it was a no-op -- it short
circuits on `isinstance(x, DTensor)` -- which is exactly why the measurement came
back bit-identical (1.0649 with and without). That machinery has been removed
rather than left as dead code that can never fire.

**The localization was not sound either.** Gate-on scored 1.0649 and gate-off
0.9801, and I read that as isolating the gate. It does not: disabling the gate
removes a whole projection (~88M parameters at K3 scale) from the model, so the
two arms are DIFFERENT MODELS with different gradient fields. A ratio that moves
when you delete a large module is not evidence that the module's TP handling is
wrong. I noted this concern in passing and then built on the comparison anyway.

## Where the TP question actually stands

Open. Established and still standing:

* `clip_grad_norm_` reports exactly the materialized norm in every configuration,
  so nothing is a reporting defect;
* the whole-model gap is real and one-directional (dp2 8.496 vs dp2xtp2 2.743,
  ratio 3.10, holding dp degree fixed);
* it compounds with depth (layer 20 = 1.066, layer 0 = 3.613);
* upstream llama3 under the identical measurement is 0.9915, so TP is not broadly
  broken in this trainer;
* KDA alone at one layer is 0.9766, i.e. clean.

Not established: which module. Every "which module" answer so far came from
comparing configurations that differ by more than the module under test.

A sound next instrument has to hold the model FIXED and vary only the
parallelism, then attribute per-parameter rather than by ablation -- for example
compare a single `KimiMLAAttention` module with and without
`apply_tp_kimi_linear` on identical weights and inputs, which is what the earlier
hand-built probe did for the CP path (and which returned bit-exact there). Ablating
modules changes the model and cannot answer it.

## The rewrite has no target: our sharding plan already IS the official one

Instruction was to tear down the TP wrapper and re-port it from official K3. Doing
that first requires knowing what the official plan is, and the answer is that ours
already matches it.

vLLM's K3 (PR #50000, `vllm/models/kimi_k3/nvidia/mla.py`) assigns:

| module | vLLM | ours |
| --- | --- | --- |
| `q_proj` (no q-lora) | ColumnParallelLinear | ColwiseParallel |
| `q_a_proj` / `q_a_layernorm` | ReplicatedLinear / RMSNorm | NoParallel |
| `q_b_proj` | ColumnParallelLinear | ColwiseParallel |
| `kv_a_proj_with_mqa` | **ReplicatedLinear** | NoParallel |
| `kv_a_layernorm` | RMSNorm (replicated) | NoParallel |
| `kv_b_proj` | ColumnParallelLinear | ColwiseParallel |
| `g_proj` (output gate) | ColumnParallelLinear | ColwiseParallel |
| `o_proj` | RowParallelLinear | RowwiseParallel |

Every assignment agrees. Re-porting would produce the same file. And it could not
have addressed the gradient question anyway: vLLM is inference-only, so the
official sources contain **no backward path for this module at all** -- which is
why there was never a reference to check the gradient against, and why this has
been hard.

What is genuinely torchtitan-specific, and therefore has no official counterpart,
is the DTensor boundary policy: every module here converts to plain tensors at its
edge (`use_local_output=True`) so PP send/recv, AttnRes block stacking and the fla
triton kernels never see a mixed-mesh tensor. That policy is ours, it is not
something to port, and it is the only place a difference could live.

## Upstream MLA control: same phenomenon, smaller

`deepseek_v3_debugmodel` has the same MLA (with q_lora_rank=0 and no output gate),
6 layers, and is upstream-validated. Under the identical instrument:

    deepseek_v3 (6 layers)   dp2 3.783382  dp2xtp2 3.658649  ratio 1.0341
    -> per layer 1.0341^(1/6) = 1.0056, i.e. 0.56% per layer

    k3mini, 1 dense MLA layer                              ratio 1.0649
    -> 6.5% in a single layer

So a known-good upstream MLA shows the SAME KIND of deviation, about 12x smaller
per layer. Two readings are consistent with that and this measurement cannot
separate them: either some deviation is intrinsic to MLA under TP and ours is
worse for a reason still unidentified, or the two models differ enough (ours has
the output gate and AttnRes; ours has 4 heads to DSv3's 16, so 2 per rank against
8) that the per-layer numbers are not comparable.

Note the same weakness as the retracted gate ablation applies here: DSv3 versus
k3mini is a comparison between different models, not an isolation of one
mechanism. It raises the suspicion again without establishing it.

## Position

No rewrite performed. The thing the instruction named as the source -- the official
TP plan -- is already what we have, so a rewrite would be cosmetic and would not
be validatable as an improvement against any baseline. The open question is the
backward boundary policy, which is torchtitan-specific and has no official
reference, and which no measurement so far has isolated.

## ISOLATED: the AttnRes projections, and they scale with TP degree

The instrument that finally worked holds the MODEL fixed (one dense MLA layer, no
KDA, no MoE) and varies only the TP degree, attributing per parameter instead of
by ablation. Gradient norms, dp2 as reference:

| parameter | dp2 | dp2xtp2 | dp2/tp2 | dp2xtp4 | dp2/tp4 |
| --- | --- | --- | --- | --- | --- |
| `final_attn_res_proj.weight` | 0.2774 | 0.6076 | **0.4565** | 0.8393 | **0.3305** |
| `mlp_res_proj.weight` | 0.5554 | 0.6323 | **0.8784** | 1.4858 | **0.3738** |
| `self_attn.attn_gate_proj.weight` | 0.3826 | 0.3750 | 1.0203 | 0.3688 | 1.0374 |
| `self_attn.o_proj.weight` | 4.2243 | 3.8554 | 1.0957 | 4.0976 | 1.0309 |

The AttnRes pseudo-query projections receive gradients that GROW with the TP
degree. Every other parameter stays flat as tp goes 2 -> 4. That is a signature no
previous measurement produced, and it comes from the only instrument so far that
neither deletes a module nor compares two different models.

Two conclusions follow immediately.

**The output gate is exonerated.** `attn_gate_proj` sits at 1.02 and 1.04 across
both degrees -- flat. The earlier gate ablation was misleading exactly as the
retraction said, and the retraction was right.

**The suspect is `block_attn_res`.** It is the one place in this model with
hand-rolled backward grad placements: it reads `proj.weight` directly rather than
calling the module, and requests `Partial()` on the tp axis to force an all-reduce
that a bare `to_local()` would skip. A gradient that grows with tp degree is what
over-reduction looks like -- the contribution is being summed across ranks that
each already hold the full value.

The scaling is not a clean 1/tp (0.4565 and 0.3305 rather than 0.5 and 0.25), so
it is not a single uniform double-count. That is expected for a NORM: these
tensors' gradients are part correct and part over-counted, and the norm mixes
both. The direction and the monotonicity in tp are the signal.

This also explains the whole-model picture without any other mechanism: AttnRes
runs in EVERY layer, its error grows with tp, and 21 layers compound it -- which is
why the end-to-end ratio was 3.10 and why it looked like it lived everywhere.

### Not yet done

The fix is not written. `block_attn_res`'s placement logic needs to be re-derived
rather than patched by guess, and it has to be verified on this same instrument:
the AttnRes rows must go flat across tp2 and tp4 like `o_proj` and
`attn_gate_proj` already do. Note that this code was previously changed to fix a
DIFFERENT symptom (grad_norm 40k-80k under a 4D mesh), so whatever replaces it has
to keep that case correct too -- the earlier fix is why the `Partial()` request is
there at all.

## Correction: AttnRes is one component, not the whole cause

The section above says the AttnRes finding "explains the whole-model picture
without any other mechanism". That is an overstatement, and a measurement already
in hand contradicts it: `kimi_linear_k3mini_diag_1l_mla_noattnres`, one dense MLA
layer with AttnRes DISABLED, gives ratio **1.0468** against 1.0649 with it. If
AttnRes were the whole cause, disabling it would land near 1.0.

The per-parameter table shows the same thing once read carefully. Calling
`o_proj` "flat" was about its behaviour ACROSS tp degrees (1.0957 -> 1.0309, it
does not grow), not about it being unbiased -- it carries ~5-10% like most
parameters:

    post_attention_layernorm 1.1982    ffn.gate_proj        1.1232
    input_layernorm          1.1537    kv_a_proj_with_mqa   1.1042
    norm                     1.1479    o_proj               1.0957

So there are TWO distinct phenomena:

1. **Grows with tp degree** -- only the AttnRes projections. That localization
   stands, and it is the one thing the fixed-model / varying-tp instrument
   isolated.
2. **A flat ~5-10% deviation on nearly every parameter**, which does NOT grow with
   tp and is still unexplained. Upstream deepseek_v3 shows 0.56% per layer for
   comparison, so ours is larger, but that comparison is between different models.

Fixing block_attn_res would address (1) and leave (2) untouched.

## 2026-07-30: AttnRes fixed; remaining gap isolated to MoE

### A measurement flaw invalidated the two entries above

Both the "AttnRes grows with tp" table and the "no-AttnRes is 1.0468" figure were
run WITHOUT a shared seed checkpoint. FSDP2 meta-init gives each parallel layout
its own RNG stream, so dp2 and dp2xtp2 started from DIFFERENT WEIGHTS -- those
runs compared models, not parallelisms. The matrix scripts already knew this
(run_fsdp_pp_cp_ep_matrix.sh seeds explicitly and says why); the ad-hoc probe
runs did not.

A second harness hole: the probe legs passed no --dump-folder, so after the first
leg wrote ./outputs/checkpoint/step-3 the later legs RESUMED FROM IT, ignored
--checkpoint.initial-load-path, and exited without running a step. The main
matrix scripts pass --dump-folder per leg and were never exposed.

Both fixed in run_tp_perparam_noattnres.sh (shared seed + per-leg dump folder).
Run-to-run noise floor of this model is 0.0000 -- bit-deterministic.

### The AttnRes defect, measured properly

One dense MLA layer, shared seed, varying only tp. Ratio = dp2 / dp2xTP:

                            before          after
                          tp2     tp4     tp2     tp4
  final_attn_res_proj   0.4999  0.2501  0.9998  1.0004
  mlp_res_proj          0.5008  0.2505  1.0016  1.0018
  (18 other params)     1.0000  1.0000  1.0000  1.0000

Exactly 1/tp, not the fuzzy trend the unseeded runs showed. grad_query is
sum_{n,b,t} grad_logits*K with both factors replicated on tp, so every rank
already holds the full gradient and Partial() summed tp identical copies. Fixed
by using the default to_local() placement.

Regression gate for the case Partial() was originally added for (grad_norm
40k-80k on a 4D mesh): fsdp2 x tp2 x cp2, full k3mini, 8 GPU -> grad_norm 3.74 /
3.82 / 3.78 over three steps. No regression. That symptom was really fixed by
local_output_grad_placements=(Replicate(),) in parallelize.py; the Partial()
request was a redundant second fix that overshot.

### What is still wrong: MoE, not KDA, not MLA, not AttnRes

Single-layer isolation, shared seed, per-parameter, max |ratio-1| over all params:

  diag_1l_mla            (MLA only)        0.0004    clean
  diag_1l_mla_noattnres  (no AttnRes)      0.0002    clean
  diag_1l_kda            (KDA only)        0.0010    clean
  diag_1l_mla_moe        (MLA + MoE)       1.4650    BROKEN

The MoE leg deviates on EVERY parameter and grows with tp. Direction: gradients
are too SMALL under tp (the mirror of the AttnRes bug -- an all-reduce missing
rather than doubled). Forward is fine: step-1 loss is 7.77043 / 7.77072 / 7.77151
at tp 1/2/4, a bf16-level spread. grad_norm at the same step is 8.8086 / 7.3616 /
6.7362.

Upstream control, same common MoE, deepseek_v3_debugmodel, tp1 vs tp2 grad_norm:
3.7201 vs 3.6052 (3.2%). Ours at tp2 is 19.7%. Upstream is not gradient-exact
under tp either, but ours is ~6x worse, so most of our gap is our own.

Lead, not yet confirmed: apply_ep_kimi_linear runs when tp > 1 even with ep == 1
(parallelize.py:206), and moe.parallelize() wires the tp mesh into the token
dispatcher. If tokens are split across tp, each rank's routed-expert gradient is
Partial on tp -- but the experts are distributed as Replicate() and
GroupedExperts.forward calls a bare to_local(), which skips that all-reduce.
Needs to be verified the same way the AttnRes one was, not assumed.

## The MoE defect, localized to the experts' input-side backward

The earlier lead in this file -- "routed experts are Replicate() on tp and
GroupedExperts.forward calls a bare to_local(), so the expert weight gradients
skip an all-reduce" -- is DISPROVEN. The expert weight gradients are exact:

  w1_EFD  0.9996 / 0.9998      shared_experts.gate_proj  0.9998 / 0.9999
  w2_EDF  0.9997 / 1.0005      shared_experts.up_proj    0.9998 / 1.0000
  w3_EFD  0.9997 / 1.0001      shared_experts.down_proj  1.0000 / 0.9999

(ratio dp2/tp2 and dp2/tp4, diag_1l_mla_moe, shared seed.)

What is wrong is the gradient flowing back OUT of the experts. Ordering the
parameters along the backward path makes the boundary exact -- Eq. 11's chain is
down -> experts -> RMSNorm -> up:

  above the experts   lm_head              0.9997 / 1.0000   exact
                      norm                 0.9995 / 1.0004   exact
                      ffn.latent.up        0.9996 / 1.0004   exact
  ---- routed experts ----
  below the experts   ffn.latent.down      1.4643 / 2.1692   wrong
                      router.gate          0.7612 / 1.3317   wrong
                      post_attn_layernorm  1.4181 / 1.8639   wrong
                      self_attn.o_proj     1.3164 / 1.6100   wrong
                      embed_tokens         1.4197 / 1.6476   wrong

Everything the backward reaches BEFORE the experts is exact; everything it
reaches AFTER them is wrong. The forward is fine (step-1 loss 7.77043 / 7.77072 /
7.77151 at tp 1/2/4), so this is purely a backward-placement defect, and it is
one edge: the gradient w.r.t. the experts' input.

Mechanism, consistent with the above but NOT yet verified: the token dispatcher
treats the tp axis as sequence-parallel -- `wire_meshes` sets `sp_size =
tp_mesh.size()` and `sp_rank = tp_mesh._sym_get_coordinate(0)`
(models/common/token_dispatcher.py:212). So each tp rank runs the experts on 1/tp
of the tokens. The forward combine reassembles them correctly. The dispatch's
backward has to reduce each rank's contribution back onto the full token set; if
it does not, the gradient arriving at routed_input carries only one rank's share,
which is exactly what the table shows. Verify before fixing, the way the AttnRes
one was.

Upstream control on the same common MoE (deepseek_v3_debugmodel, tp1 vs tp2
grad_norm): 3.7201 vs 3.6052, a 3.2% gap against our 19.7%. Upstream is not
gradient-exact under tp either, so part of this may be shared with upstream
rather than specific to the K3 wiring.

### The sequence-parallel mechanism is DISPROVEN

The mechanism proposed above -- dispatcher treats tp as sequence-parallel, each
rank runs the experts on 1/tp of the tokens, dispatch's backward fails to reduce
-- is wrong. Two measurements:

Runtime check confirms the wiring is real: the dispatcher IS
`AllToAllTokenDispatcher` and `wire_meshes` does set `sp_size = 2` under tp2.

But suppressing it changes nothing. Passing `tp_mesh=None` into `wire_meshes`
(so `sp_size` stays 1) reproduces the defect exactly:

              tp1       tp2       tp4
  with SP    8.8086    7.3616    6.7362
  no SP      8.8086    7.3616    6.7362     grad_norm, step 1

Bit-identical. The dispatcher's tp wiring has no bearing on this defect.

### What the numbers do say

The deviating parameters scale as ~sqrt(tp), not 1/tp or tp. Median ratio
divided by sqrt(tp): 0.987 at tp2 (individual params 0.97-1.04) and 0.865 at tp4
(0.79-1.23). A sqrt(tp) norm ratio is what you get when the true gradient is a
sum of tp near-orthogonal contributions and only one of them survives -- the
signature of a missing reduction, not of a wrong scale factor. The looser fit at
tp4 is consistent with the shares being partially correlated rather than
independent.

Still true and still unexplained: expert weight gradients are exact, every
parameter the backward reaches before the experts is exact, every parameter it
reaches after them is wrong, and the forward matches. The defect is one backward
edge -- the gradient w.r.t. the experts' input -- and neither the expert
to_local() nor the dispatcher's tp wiring is responsible.

Next candidate, unverified: `no_par_local = NoParallel(use_local_output=True)`
(parallelize.py:516) does not actually pass `local_output_grad_placements`,
despite the module docstring at line 453 stating that every NoParallel call
passes `(Replicate(),)`. Whatever the default resolves to is what governs the
plain-tensor -> DTensor conversion at exactly the boundary where this defect
lives. Check what the default is before changing it.

### It is pure TP, and the boundary is one tensor wide

FSDP is not involved. Dropping dp_shard to 1 reproduces the defect at the same
magnitude (grad_norm 8.2308 / 6.8119 / 6.2441 at tp 1/2/4, ratios 1.208 and
1.318, against 1.197 and 1.308 with dp2). Without FSDP in the picture the
per-parameter boundary is exact:

  lm_head                     1.0000  1.0000
  norm                        1.0001  1.0000
  ffn.latent.up               1.0000  1.0000
  routed_experts.w1_EFD       1.0000  1.0001
  routed_experts.w2_EDF       1.0000  1.0000
  shared_experts.down_proj    1.0000  1.0000
  ---- corruption enters here ----
  ffn.latent.down             1.4646  2.0702
  ffn._moe.router.gate        1.3941  1.6901
  self_attn.o_proj            1.3315  1.6317
  embed_tokens                1.4719  1.7499

Median ratio/sqrt(tp): 1.008 at tp2, 0.887 at tp4.

Eq. 11's chain is down -> experts -> RMSNorm -> up, so in the backward the
experts' own weight gradients are computed correctly (1.0000) and the very next
thing -- the gradient w.r.t. their INPUT, which is latent.down's output gradient
-- is already wrong. The defect is one tensor wide.

Also ruled out on the way: NoParallel does not accept
``local_output_grad_placements`` at all. ``_prepare_output_fn`` ends in a bare
``outputs.to_local()`` (distributed/tensor_parallel.py:82), so the backward
placement defaults to the output layout, Replicate. The claim in
kimi_k3/parallelize.py:453 that "every NoParallel call passes
local_output_grad_placements=(Replicate(),)" describes an API that does not
exist; that docstring needs correcting regardless of this bug.

The open puzzle: under TP the whole MoE subtree is replicated (router.gate,
latent, shared_experts are NoParallel; routed experts are DTensor(Replicate)
to_local'd to plain), so every tp rank should compute an identical forward AND an
identical backward, and no parameter should move at all. Something in that
subtree is nonetheless rank-dependent. Note the forward is not quite bit-identical
either -- loss 7.79058 / 7.79053 / 7.79052 at tp 1/2/4, whereas the PP legs were
bit-identical at 7.71304 -- so a small rank-dependence is visible in the forward
too, and the gradient effect is far larger than that difference would explain.

Next step: instrument the gradient w.r.t. routed_input (the tensor between
latent.down and the experts) and compare it BOTH across tp ranks and against tp1.
That distinguishes "the ranks disagree" from "the ranks agree but a reduction is
dropped", which the parameter-level view cannot separate.

## 2026-07-31: the MoE defect was two bugs, one ours and one upstream's

### Ours: the MoE latent input's gradient was declared Replicate

`moe_edge_grad_probe.py` settled what the parameter-level view could not -- it
hooks the tensor between latent.down and the experts and reports its gradient on
EVERY rank, so "the ranks disagree" and "the ranks agree but a reduction is
dropped" look different:

  tp1                                          0.04075606
  tp2   rank0 0.02921830, rank1 0.02839850     ranks DISAGREE
  tp2   after all-reduce                       0.04075651   = tp1
  tp4   after all-reduce                       0.04075698   = tp1

The shares sum to the tp1 value, so the gradient is genuinely Partial and
NoParallel's bare `to_local()` was declaring it Replicate, keeping one share.
`_NoParallelPartialGrad` on `ffn.latent.down` fixes it:

           tp1      tp2      tp4
  before  8.2308   6.8119   6.2441     19.7% and 32.1% off
  after   8.2308   8.2169   8.2110     0.17% and 0.24% off

Applied to `down` only. `up` and `norm` are on the far side of the experts and
receive a genuinely replicated gradient; declaring Partial there would
over-reduce by exactly tp -- the block_attn_res failure mode.

### Upstream's: the router gate, reproduced on unmodified deepseek_v3

The one parameter the fix did not move is `ffn._moe.router.gate` (1.3941 / 1.6901).
Its TP-plan entry never applies -- the module-internal MoE parallelization sets
`ffn` to None and leaves the whole `_moe` subtree out of the plan, so the gate is
owned by `models/common/moe_sharding.py::_router_gate_config`.

This is not K3-specific. Unmodified `deepseek_v3_debugmodel`, dp1, seeded, ratio
dp1/tp2 per parameter -- its five router gates are its five worst parameters:

  layers.2 moe.router.gate.weight   1.4780
  layers.4 moe.router.gate.weight   1.4401
  layers.3 moe.router.gate.weight   1.4178
  layers.5 moe.router.gate.weight   1.2109
  layers.1 moe.router.gate.weight   1.2067
  (next worst, ffn_norm)            1.0178

All near sqrt(2), the same signature and the same magnitude as ours. Every other
upstream parameter is within 1.8%, matching our post-fix state.

Note `_router_gate_config`'s own docstring header in moe_sharding.py:238 reads
"Router gate: dense-family TP plan with Partial output grad", but the EP-off
branch sets out_src/out_dst to `dense_activation_placement(tp=spmd.R)` --
Replicate. The intent recorded in the comment and the placement actually declared
disagree, which is consistent with the measurement.

Not patched here: this is `torchtitan/models/common/`, core code shared by every
MoE model, and the repo rule is not to modify core for an experiment. It should
go upstream as a bug report with the deepseek_v3 reproduction above, which needs
none of our code.

## After the root-cause fix: where the residual sits

Single-module re-verification with the current code (shared seed, dp2 held fixed,
only tp varies), max |ratio-1| over all parameters:

  diag_1l_mla       0.0018    clean
  diag_1l_kda       0.0010    clean
  diag_1l_mla_moe   0.018     router.gate 0.9879 / 0.9820, the rest under 0.007

MoE went from 1.465 to 0.018. KDA is and was clean, so the large A_log ratios in
the 21-layer model are not a KDA module defect -- A_log's gradient is tiny
(largest 0.11% of the total norm) and its ratio is dominated by near-zero values.

Full K3, 21 layers, dp2 fixed: median |ratio-1| 0.80% at tp2, 0.86% at tp4,
grad_norm 11.7447 / 11.7856 / 12.0221.

The largest single-module residual is the router gate at 1.2-1.8%. Upstream
deepseek_v3 after the same fix sits at 0.25% over 83 parameters, so 1.8% is
unlikely to be bf16 reordering alone.

Next candidate, IDENTIFIED BUT NOT VERIFIED: K3 calls the MoE as
``self._moe(self.latent.to_latent(x), router_input_BLD=x)`` -- the router reads
``x`` while the experts consume ``W_down x``, so they are different tensors. The
MoE-level sharding config (moe_sharding.py:187) declares only ``x_BLD``;
``router_input_BLD`` appears in no sharding config at all, so its placement and
its gradient placement are undeclared on the path K3 uses. That is the same class
of gap as the one just fixed, on the one input that only K3 supplies -- which
would also explain why the residual is larger for us than for deepseek_v3, which
never passes it. Verify on diag_1l_mla_moe before changing anything.

### router_input_BLD is DISPROVEN

The undeclared second input is not the cause. Hooking it directly:

  tp1                                      0.01914334
  tp2   rank0 0.01914365, rank1 0.01914365

The ranks agree with each other and match tp1 to 1.6e-5 relative. So even though
``router_input_BLD`` appears in no sharding config, its gradient comes back
correct on the path K3 uses. (The declaration gap is still real and worth closing
for clarity, but it is not this bug.)

That narrows the router gate's remaining 1.2-1.8% to its OUTPUT side.
``grad_W_gate = grad_scores^T @ router_input``, and the right factor is now
verified clean, so the error is in ``grad_scores`` -- the gradient flowing back
into the gate from the MoE body, through sigmoid/softmax, expert_bias,
route_norm and route_scale before reaching the local_map region whose
in_grad_placements were just fixed.

Note the size and shape of what is left: the gate's own gradient norm is 0.1352
against a model total of 11.74, the deviation is 1.2% at tp2 and 1.8% at tp4, and
it is monotonic in tp with the tp legs slightly LARGER. Small, systematic, and on
a small-magnitude gradient passing through two nonlinearities -- consistent with
either a residual placement issue or genuine bf16 sensitivity. Distinguishing
those needs an fp32 run, which is the cheapest next measurement: if the gap
survives in fp32 it is a placement bug, if it collapses it is precision.

## The router residual is precision, not placement -- but 21 layers are not clean

Re-measuring diag_1l_mla_moe at dp1 (no FSDP, so params stay fp32) instead of
dp2 settles the precision-vs-placement question:

  max |ratio-1|   dp2: 0.018 (router.gate 0.9879 / 0.9820)
                  dp1: 0.0009 / 0.0005  (router.gate 0.9991 / 0.9995)

With FSDP's bf16 mixed-precision cast out of the picture, TP on one MLA+MoE layer
is exact. The 1.2-1.8% was bf16, introduced by the cast interacting with TP's
different accumulation order -- not a placement bug.

### But the multi-layer model is a different story

21 layers, all-MLA (diag_no_kda, chosen because the KDA shmem limit blocks pure
TP on the real flavor), dp1, pure TP, 458 parameters:

  median |ratio-1|   tp2 0.0054   tp4 0.0047
  max                tp2 0.1497   tp4 0.1932

and the worst are not negligible parameters:

  parameter                        |grad|   % of total   /tp2     /tp4
  layers.0.mlp_res_proj            1.0633     7.85%     0.9260   0.8221
  layers.1.mlp_res_proj            0.6194     4.57%     1.1095   1.1910
  layers.1.attn_res_proj           0.1906     1.41%     0.9106   0.8068
  embed_tokens                     4.4453    32.81%     0.9577   0.9782
  layers.0.kv_a_proj_with_mqa      3.8017    28.06%     0.9481   0.9797

The 42 AttnRes projections together carry 9.6% of the model's gradient norm.

### What this says about the AttnRes fix

It was verified ONLY on a single-layer instrument, and a single layer
under-exercises block_attn_res: with one block the softmax over the block axis is
nearly degenerate, so the pseudo-query gradient path the multi-layer model uses is
barely exercised. "All parameters 1.0000 at tp2 and tp4" was true of what was
measured and does not extend to 21 layers, where the AttnRes projections are the
worst weight-carrying parameters under pure TP.

The bare to_local() that replaced the Partial() request may therefore be right for
the one-block case and wrong once there are several blocks -- some contributions
to the pseudo-query gradient may genuinely be partial across tp. That has to be
measured on a multi-layer instrument, not reasoned about.

Next: build the same per-parameter instrument at 2 and 4 layers (enough for
several AttnRes blocks, small enough to read), confirm AttnRes is the cause by
comparing against diag_no_kda with AttnRes disabled, then re-derive the placement.
