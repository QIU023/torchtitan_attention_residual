# PR #6 — RS+merge+AG seq-shard fusion (documented as a feature)

**Target repo**: `sgl-project/sglang`
**Target surface**: docs + `model_runner_hooks.py` example hook.
**Fork reference**: pattern lives inside the AttnRes inference overlay
(`python/sglang/srt/layers/attn_res.py`, fused-Triton path in commit
`63325b2b4`); documentation-only here, no fork commit isolating the
fusion as a standalone feature.

**Effort**: M (design RFC + docs + 1 reference hook example).
**Risk**: low (docs + example, no kernel change).
**Track**: **RFC** — needs upstream decision on whether to bless this
as a documented feature or keep it inside per-model overlays.

---

## Suggested PR title

> [RFC] Document the RS+merge+AG seq-shard fusion pattern as a
> first-class model-runner hook

---

## Summary

In our AttnRes inference overlay, we replace the default per-layer
`AllReduce(o_proj)` with `ReduceScatter(o_proj_partial) + cross-layer
merge + AllGather`. The fusion halves on-wire AllReduce volume across
a PP / TP group by deferring the reduce + merging across cooperative
layers before the final all-gather.

The fusion is **model-agnostic** — applies to any TP=N model whose
`o_proj.reduce_results=False` AND whose model-runner can stage the
deferred reduces across cooperative layers.

## Motivation

Today the fusion is buried inside our AttnRes overlay
(`layers/attn_res.py`'s Phase-2 fused Triton path). Any other
TP-heavy model — DeepSeek-V3, Mixtral, Llama 4 — would benefit if the
pattern were documented as a feature, with a reference
`model_runner_hook` showing how to install it.

This isn't a code-change PR; it's an RFC to **promote the pattern from
"buried in one overlay" to "documented + has an example hook"** so
other model authors can adopt.

## Proposal

1. **Docs page**: `python/sglang/docs/advanced/rs_merge_ag_seq_shard_fusion.md`
   - When the pattern applies (TP=N + cooperative layers + sequence-
     parallel boundaries).
   - Algorithm sketch: deferred reduce → cross-layer merge →
     coordinated all-gather.
   - Bandwidth savings: ~50% on AllReduce volume across cooperative
     layer groups; latency reduction depends on overlap.
   - Implementation requirements: model author wires
     `o_proj.reduce_results=False` + registers cooperative-group
     barrier metadata.

2. **Reference hook**:
   `python/sglang/srt/model_executor/hooks/rs_merge_ag.py`
   - Generic helper that any model class can call from its
     `init_layers` to declare cooperative groups.
   - Tested via a smoke model that exercises the fusion on a 2-layer
     dense model.

3. **Adopter example**: the AttnRes overlay (PR #5) is the in-tree
   example after #5 lands. Docs page links to it.

## Filing checklist

- [ ] File as an RFC issue first; let maintainers decide if the
      pattern is general enough to warrant a documented feature vs
      staying inside per-model overlays.
- [ ] Wait for at least one other in-tree model to express interest
      (DeepSeek-V3 / Mixtral maintainers) — single-adopter features
      tend to bounce.
- [ ] If accepted, follow-up PR adds the docs page + reference hook.
- [ ] No production-blocking; can wait behind PR #5 (algorithm) and
      PR #4 (torchtitan signature).
