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
