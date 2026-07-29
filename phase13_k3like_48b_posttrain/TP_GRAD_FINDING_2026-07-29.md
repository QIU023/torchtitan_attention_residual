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
