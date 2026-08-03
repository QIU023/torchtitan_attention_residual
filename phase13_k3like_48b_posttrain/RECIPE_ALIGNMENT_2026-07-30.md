# Training-recipe alignment: Muon + Quantile Balancing (2026-07-30)

Structure alignment was verified module by module against the released reference
(MLA rel 7.07e-06, KDA 2.67e-03, SiTU/LatentMoE/router structurally identical,
2p8t config 29/29 fields, 497,220 checkpoint keys mapped). What remained were two
RECIPE items: K3 trains with Muon on its matrix parameters (report sec 2.5) and
with Quantile Balancing on the router (sec 2.3.3), while this repo defaulted to
AdamW and core's sign rule. Both existed in the tree with nothing selecting them.

## Measured, all four legs from the same seed checkpoint

k3mini, dp2, global batch 8, seq 512, seed 42 deterministic.

| step | AdamW + sign | Muon | QB | Muon + QB |
| --- | --- | --- | --- | --- |
| 1 | 7.71304 | 7.71304 | 7.71304 | 7.71304 |
| 2 | 7.64636 | **7.54796** | 7.65102 | **7.55319** |
| 3 | 7.50443 | **7.29315** | 7.49088 | **7.29505** |
| 4 | 7.23409 | **6.95117** | 7.23757 | **6.95279** |

Step 1 is identical in all four arms because the loss precedes the first optimizer
update -- which is the control confirming the arms share seed and data, so the
later divergence is attributable.

Muon dominates the trajectory. **QB's effect is small here and that is expected,
not a disappointment**: with 8 experts and top-2 over 4 steps there is almost no
load imbalance for a balancing rule to correct, so the solved bias and the sign
rule land in nearly the same place. QB's effect is demonstrable where imbalance
exists -- the unit tests drive a deliberately skewed router from load cv 0.607 to
0.147 -- and at K3's 896 experts it is the whole point. Claiming a training win
for QB from this table would be reading noise.

## Where the recipe is applied, and why not everywhere

`kimi_k3_mini_k3recipe` carries both. Deliberately a separate flavor rather
than a change to `kimi_k3_mini_block_attn_res`, because that flavor carries
the cross-parallelism numerical baselines (PARALLEL_NUMERIC_BASELINE,
PP_VP_REEXAMINATION) and changing its optimizer or router rule would invalidate
every recorded number. The baseline flavor stays a fixed reference.

The 2p8t flavor was deliberately NOT given the recipe either: it cannot run on
this hardware, so adding config there would be shipping something unexercised --
the failure mode this phase spent days removing. It should get the recipe on a
machine that can run it, and the recipe flavor above is the reference for what to
copy.

## Implementation notes worth keeping

Core's `_resolve_optimizer_cls` hardcodes `{Adam, AdamW}` and raises otherwise, so
`KimiOptimizersContainer` subclasses it. That subclass needs its OWN nested
`Config` even though it adds no fields: `Configurable` sets `_owner` per Config
CLASS, so inheriting the parent's verbatim leaves `_owner` pointing at core's
container and `build()` silently returns the wrong class. The smoke surfaced it as
"Optimizer Muon not added".

Muon tagging happens at BUILD time via a `per_head_muon` spec flag, not in
`post_optimizer_build_fn`. The optimizer is constructed from the parameters and
Muon reads `_muon_heads` off each one, so a tag applied after the optimizer exists
is invisible to it -- which is exactly how Per-Head Muon was inert for weeks.

`default_muon` lists the AdamW group FIRST because the container assigns each
parameter to the first matching pattern, and uses separate learning rates: Muon's
update is orthogonalized, so its scale is decoupled from gradient magnitude and it
wants a much larger lr. Sharing one lr is the usual way to make Muon look bad.
