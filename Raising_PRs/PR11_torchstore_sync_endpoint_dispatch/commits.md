# Backing commits — PR #11 torchstore sync-endpoint dispatch

## Discovered in

**Phase 11** — GRPO multimodal RLHF chain setup. The actor mesh's
SGLang Generator is async; torchstore's Controller endpoints used for
weight-sync / state-broadcast are sync. The first cross-mesh call
hard-failed at runtime with a sync/async mismatch error.
`phase11/rlhf/run_grpo_llava_kimi.py` installs a Controller monkey-
patch at process start (and in every Monarch-spawned subprocess) to
wrap the 5 sync endpoints as async-coroutine adapters.

## Fork source

**None.** The workaround lives in launcher code, not in torchstore
itself. The patch (`run_grpo_llava_kimi.py`'s monkey-patch) is not
what we want to upstream — it's a runtime override that should be
replaced by an upstream-side API for declaring dispatch mode.

## Status

- **Issue**: ready to file (see PR.md for issue body).
- **Patch**: depends on upstream API direction (endpoint-level
  declaration vs env-gated bulk override).

## Filing recipe

```bash
# 1. File the torchstore issue using PR.md body as the seed.
# 2. Include a minimal Python repro (strip Monarch / RL dependencies
#    down to a `Controller` + sync/async call mismatch).
# 3. WAIT for maintainer response on which API direction (endpoint-
#    level dispatch_mode declaration vs env-gated bulk override).
# 4. Once direction is decided, write the ~30-line patch against
#    torchstore main and open the PR.
```

## What the fork's workaround does (for reference in the issue body)

The Controller monkey-patch in `phase11/rlhf/run_grpo_llava_kimi.py`:

1. Imports `torchstore.Controller`.
2. For each of the 5 sync endpoints (`put`, `get`, `broadcast`,
   `barrier`, `shutdown`), wraps the existing method in an `async def`
   that awaits the underlying sync call inside a thread executor.
3. Replaces the Controller's bound methods with the wrapped versions.
4. Re-applies the same monkey-patch inside every Monarch-spawned
   subprocess (the actor mesh's worker processes), because Monarch's
   spawn doesn't inherit the patched module state.

0 performance impact (pure wrappers), but the per-subprocess
re-application is fragile.

## Notes for the PR opener

- The issue body should be **engine-agnostic** — the SGLang Generator
  / Monarch / RL specifics are repro context, but the underlying
  request is general (any async-caller library hitting torchstore
  will hit this).
- Don't propose a specific API in the issue; let maintainers pick.
  Both options are documented in PR.md.
- The fork's monkey-patch is **not** PR material as-is; it's listed
  here for the issue body to reference as "we have a working
  workaround, but it's brittle, and the upstream API should make it
  unnecessary".
