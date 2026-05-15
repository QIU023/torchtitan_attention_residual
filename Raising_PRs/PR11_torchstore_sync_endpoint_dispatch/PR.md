# PR #11 — torchstore sync-endpoint dispatch policy (async caller opt-in)

**Target repo**: `pytorch/torchstore`
**Target surface**: endpoint dispatch policy (likely
`torchstore/controller.py` + endpoint declaration API).
**Fork reference**: workaround live as a runtime Controller monkey-patch in
`phase11/rlhf/run_grpo_llava_kimi.py` (both main process and
Monarch-spawned subprocesses).
**Effort**: file the **issue first** (~1 hour); patch shape ≈ 30 lines
pending upstream API decision.
**Risk**: medium — API design choice that affects all torchstore consumers.

---

## Filing order

**File the issue first**, no patch. Let torchstore maintainers decide
between the two acceptable shapes (endpoint-level declaration vs
env-gated bulk override).

---

## Issue title

> Sync-endpoint dispatch rejects async callers: surface an opt-in for mixed-mode use

---

## Issue body (suggested)

### Problem

torchstore's `Controller` rejects an async caller hitting a sync
endpoint (and vice versa). Concrete repro: in our GRPO RL loop the
actor mesh's `Generator` is async (SGLang Engine returns a coroutine),
but torchstore's 5 endpoints used for weight-sync and state-broadcast
(`put`, `get`, `broadcast`, `barrier`, `shutdown`) are declared sync.

Result: hard exception at the first cross-mesh call. The runtime API
has no documented escape hatch.

### Repro outline

```python
import torchstore
import asyncio

class AsyncCaller:
    async def fetch_weights(self):
        # This call from an async context hits the controller's sync
        # endpoint → torchstore raises "sync endpoint called from async
        # context" or similar.
        return torchstore.controller.put(...)

asyncio.run(AsyncCaller().fetch_weights())
```

Minimal repro can be reduced further; happy to provide one matching
maintainer's preferred shape.

### Our workaround

In `phase11/rlhf/run_grpo_llava_kimi.py` we monkey-patch the
`Controller` at process start in BOTH the main process AND every
Monarch-spawned subprocess to wrap the 5 sync endpoints into thin
async-coroutine adapters. **0 performance impact** (pure wrappers),
but the patching machinery is fragile — adding a new endpoint
upstream silently breaks our wrapper.

### Same pattern as SGLang's `SGLANG_DISABLE_SHM_MM`

Upstream-side strict policy, add an opt-in flag for callers that know
what they're doing. Two viable upstream shapes:

1. **Endpoint-level declaration** (preferred): let endpoint authors
   declare `dispatch_mode={sync, async, auto}`. `auto` accepts either
   and wraps internally. Backwards-compatible (default `sync` keeps
   today's behaviour).
2. **Env-gated bulk override**: `TORCHSTORE_ALLOW_MIXED_SYNC_ASYNC=1`
   relaxes the dispatch policy globally. Lighter to land, less clean
   long-term.

### Why this isn't trivially landable

API design choice — pure dispatch-mode-on-endpoint is cleaner but
more invasive; env flag is simpler but lasting (env-config debt).
Upstream call on which direction to take. ~30 line patch once
direction is decided.

### Filing checklist

- [ ] File the torchstore issue using this body as a starting point.
- [ ] Include a minimal Python repro (no Monarch / RL dependencies).
- [ ] WAIT for maintainer response on API direction.
- [ ] Once direction is chosen, open the patch PR.
