# RFC question to ask before the CP and PP axis PRs

**Status**: draft, not posted. Needs a go-ahead before it goes anywhere upstream.

**Why ask first**: the answer decides whether the CP and PP PRs ship as written
(imperative wiring) or get rewritten declaratively. Rewriting after review is the
expensive order.

## What we found that changes the question

`apply_cp_to_forward` already carries this, in its own docstring:

> TODO: This is a temporary workaround that manually allgathers K/V (FlexAttention) or
> wraps inputs as CP-sharded DTensors (SDPA). Once all models adopt config-based sharding
> with full DTensor, CP redistribution should be expressed declaratively via
> ShardingConfig and this function should be removed.

So "can `ShardingConfig` express CP" is already answered in the intended direction, and
the question we should actually ask is narrower: **does the declarative target cover a CP
that is not a redistribution?**

The distinction matters because the two halves of our CP are not the same kind of thing:

- **MLA layers, Ulysses.** This one *is* a redistribution -- one all-to-all taking a
  sequence-sharded activation to a head-sharded one. A placement pair (`Shard(seq)` ->
  `Shard(head)` on the cp axis) describes it, and it looks like it would fit the target
  the TODO describes.
- **KDA layers, KCP.** This one is not. The sequence stays sharded on every rank and the
  recurrence is recomputed: a prefix scan composes each rank's (cumulative transition,
  zero-started state) fragment to recover its true incoming state, and a separate
  fixed-size halo covers the short convolutions. No placement of the module's tensors
  says any of that -- the arithmetic inside the module changes, and the two cross-rank
  exchanges have different shapes from each other. It is a different forward, not a
  differently-placed one.

## The question, as one paragraph

Is the declarative CP target in `apply_cp_to_forward`'s TODO meant to cover only
redistribution-shaped CP (Ulysses, ring), with algorithmic CP staying module-internal?
Linear-attention CP is a state-passing scan, so it has no placement formulation; if the
answer is "declarative eventually covers everything", we would want to know the intended
shape before writing the K3 CP against an interface that is about to change.

## The same question for PP, which is a separate one

`ShardingConfig`'s six fields are all placements of a module's own parameters and
activations. Splitting a module across pipeline stages is a change to the module graph,
not a placement, so we assume PP is out of scope for config-based sharding entirely and
stays imperative. Worth confirming, because our PP carries a cross-stage adapter and we
would rather hear "yes, imperative" now than at review.

## What we do while waiting

Ship both PRs imperative, matching what upstream's own CP does today, and say in each PR
that it follows `apply_cp_to_forward`'s current shape and will move when that TODO does.
That is the smaller claim and it does not block on an answer.
