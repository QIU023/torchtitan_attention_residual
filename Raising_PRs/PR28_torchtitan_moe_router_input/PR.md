# PR28 -- torchtitan: let MoE route on a different tensor than the experts consume

**Target**: `pytorch/torchtitan`, `torchtitan/models/common/moe.py` +
`torchtitan/models/common/moe_sharding.py`.
**Independent of PR-4025** and of PR27, though both touch `common/moe.py`; the two
patches apply to `main` in either order and do not overlap.

**Status**: not filed. Patches: `router_input.patch` (the forward parameter, cut from
the current tree) and `sharding_entry.patch` (the `moe_sharding.py` half, see below).

## PASTE (PR description)

`MoE.forward` feeds the same tensor to the router and to the experts. This adds a
keyword-only `router_input_BLD` that, when given, is what the router reads; the experts
still consume `x_BLD`. Default None, so the conventional MoE is unchanged.

Latent-expert designs need the two to differ: the router scores the full-width token
while the experts receive a narrower projection of it. Only the leading `(B, L)` has to
match.

Keyword-only is deliberate. `Module._cache_pos_arg_names` snapshots the positional
parameters of `forward`, and `LocalMapConfig.in_grad_placements` is a tuple ordered by
that list, so a positional parameter here would perturb the mapping for any module that
later grows a `local_map`. Keyword-only args are excluded from that list.

The second half is the sharding entry. `_moe_sharding_config` keys `in_src_shardings` /
`in_dst_shardings` by parameter name, so an input with no entry is passed through
unredistributed -- `_redistribute_inputs` skips names it does not find. Without naming
`router_input_BLD` there, a caller passing a Replicate activation reaches a router whose
gate declares SP, and nothing raises; it just routes on the wrong placement. The entry
mirrors `x_BLD`, which is right because both are `(B, L, *)` activations, and the None
default is skipped by the existing `isinstance(value, torch.Tensor)` guard rather than
needing a new one.

We hit exactly that: our model was patching the config with `dataclasses.replace` after
the fact to name the second input.

## Evidence

- The parallelism matrix's latent-MoE cells train with the router reading the pre-latent
  activation, across TP/SP layouts (the case the missing sharding entry silently broke).
- A stock model is unaffected: `router_input_BLD` is None, so both the branch in
  `forward` and the config entry are no-ops. See PR27's `RESULTS.md` for the
  before/after identity run, which covers this patch too.

## Held back

`sharding_entry.patch` is currently NOT applied to our tree -- it was cut and set aside
so a running 58-cell gate would not see the file change mid-run. Re-apply and re-run the
latent cells before filing.
