# PR #11 — filing instructions (issue-first)

## Status

🟠 **Issue ready to file; patch deferred pending upstream API decision.**

## Where to file

| Repo | URL | What to file |
|---|---|---|
| pytorch/torchstore | https://github.com/pytorch/torchstore/issues/new | The issue body |

(torchstore lives at github.com/pytorch/torchstore. If by the time of
filing the project has migrated to a different namespace, retarget.)

## Title

```
[controller] Sync-endpoint dispatch rejects async callers — surface an opt-in for mixed-mode use
```

## Body

Use [PR.md](PR.md) → "Issue body (suggested)" section verbatim. Already
contains:

- Concrete problem (async caller → sync endpoint hard exception)
- Repro outline (minimal Python; will tighten for the maintainer ask)
- Our workaround (Controller monkey-patch in main + Monarch-spawned
  subprocesses; wraps 5 sync endpoints into async coroutine adapters)
- Why it's NOT trivially landable (API design choice between
  endpoint-level declaration vs env-gated bulk override)
- Two viable upstream shapes, recommended (1) endpoint-level
  `dispatch_mode={sync,async,auto}` declaration

## What we are NOT filing yet

The patch itself. The patch shape depends entirely on which API
direction the maintainers pick:

- **Endpoint-level declaration**: ~30 lines in `controller.py` plus
  decorator/metadata changes on each `@endpoint`.
- **Env-gated bulk override**: ~10 lines in `controller.py`, single
  env-var read.

We file the patch after the API direction is acknowledged.

## Fork-side workaround reference

Live as a runtime Controller monkey-patch in
[phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py](../../phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py)
applied at process start in BOTH the main process AND every
Monarch-spawned subprocess. Zero performance impact (pure async
coroutine wrappers), but fragile to upstream API additions.

## Cross-link with other PRs in this batch

- **PR #4 (OBSOLETED)** — the original signature-stability fix in
  torchtitan's RL trainer was the sibling-API-design class of problem;
  upstream `627f4a31` resolved that one. PR #11 is the equivalent for
  torchstore's dispatch surface.
- **PR #12** (engine-agnostic Generator) — the GRPO control plane that
  exposes the async/sync mismatch. PR #11 unblocks the cleaner Generator
  shape (no monkey-patch in the launcher).
