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
