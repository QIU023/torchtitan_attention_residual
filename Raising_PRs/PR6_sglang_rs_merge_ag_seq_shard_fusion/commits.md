# Backing commits — PR #6 RS+merge+AG seq-shard fusion

## Discovered in

**Phase 11** — surfaced while implementing the Block AttnRes inference
overlay (PR #5). The fusion pattern is embedded in
`layers/attn_res.py`'s two-phase Triton kernel; we observed it could
be lifted out as a model-agnostic optimisation but didn't refactor it
into a standalone feature for this fork.

## Fork source

| Commit | Where it lives | What's relevant |
|---|---|---|
| `63325b2b4` | `python/sglang/srt/layers/attn_res.py` | fused Triton Phase-2 merge with deferred reduce + cooperative all-gather |
| `2f2e917d8` | `python/sglang/srt/layers/attn_res.py` | original two-phase + seq-shard skeleton |

All on `QIU023/sglang@attention_residual_inference` (and `main` after
the merge to `dc154e785`).

## Status

**No isolated fork commit.** The fusion is implementation detail of
the AttnRes overlay (PR #5). To file PR #6 as a standalone feature
would require:

1. Extracting the deferred-reduce + cooperative-AG plumbing from the
   AttnRes overlay into a generic `model_executor/hooks/rs_merge_ag.py`.
2. Writing the docs page + reference hook example.
3. Adding a smoke model that exercises the fusion without AttnRes.

This is **RFC-track**: file an issue first, only do the extraction
work after maintainer green-light.

## Filing recipe

```bash
# 1. File the RFC issue using PR.md as the body.
# 2. WAIT for maintainer response on whether to bless the pattern.
# 3. If accepted:
#    - Extract the deferred-reduce + AG plumbing from the AttnRes
#      overlay (commits 63325b2b4 + 2f2e917d8 in our fork).
#    - Write the docs page + reference hook example.
#    - Open PR.
# 4. If rejected (or "stays inside overlays"): close the issue, leave
#    the fusion buried in PR #5's overlay as-is. No further work.
```

## Notes for the PR opener

- This is a **post-PR-#5** filing. PR #5 (the overlay itself) is the
  natural in-tree adopter — without it, there's no concrete model
  using the fusion to point maintainers at.
- The "adopters" angle is what makes or breaks this. If only the
  AttnRes overlay uses it, maintainers will (correctly) say "fold
  back into PR #5 and skip the standalone feature".
- Watch DeepSeek-V3 / Mixtral upstream PRs for similar fusion needs;
  if a second adopter emerges, ping the maintainers.
