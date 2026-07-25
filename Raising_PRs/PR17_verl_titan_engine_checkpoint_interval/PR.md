# PR #17 — verl torchtitan engine silently never saves model weights

**Target repo**: `volcengine/verl`
**Target file**: `verl/workers/engine/torchtitan/transformer_impl.py` (`TorchTitanEngine.__init__`, the `CheckpointManager.Config` construction)
**Fork reference**: verl `kimi_k3_integration`, commit `c2c2c5a8`
**Upstream audit (2026-07-25)**: still present on `upstream/main` @ `983cb0f2` — the engine builds `CheckpointManager.Config(...)` with no `interval`, so torchtitan's default applies. **Not obsoleted; still worth filing.**
**Effort**: ~1 hour (one-line change + a short-run save test).
**Risk**: very low — the engine already gates saving on verl's own `save_freq`; this only stops torchtitan from second-guessing that decision.

---

## Suggested PR title

> [engine] fix: torchtitan engine never writes model weights when `save_freq` is not a multiple of torchtitan's checkpoint interval

## Suggested PR body

### Summary

`TorchTitanEngine` constructs its `CheckpointManager.Config` without an
`interval`, inheriting torchtitan's default. `save_checkpoint()` also does not
pass `last_step`. torchtitan's `CheckpointManager.save()` is therefore free to
decide the call is not on a checkpoint boundary and write only the pieces it
saves unconditionally — so a verl run whose `trainer.save_freq` does not land
on a multiple of that default writes **no model weights at all**, including at
the final step. There is no error and no warning; the checkpoint directory
exists and contains dataloader state, so the run looks successful until someone
tries to load the model.

There are two cadence authorities in this path and they disagree. verl's
`trainer.save_freq` already decides when to save (the engine's
`save_checkpoint()` is only called on those steps), so torchtitan's interval is
redundant here — and when it disagrees it wins silently.

### Fix

Set `interval=1` in the engine's `CheckpointManager.Config`: every `save()`
call verl issues then goes through torchtitan's normal
model + optimizer + extra-state DCP path. verl keeps sole ownership of the
cadence.

```python
checkpoint = CheckpointManager.Config(
    enable=True,
    initial_load_in_hf=True,
    initial_load_model_only=True,
    initial_load_path=model_config.path,
    # verl's trainer.save_freq is the cadence authority and save() is only
    # called on those steps; defer torchtitan's own interval to 1 so every
    # requested save actually writes.
    interval=1,
)
```

### Impact

Any `verl` SFT/RL run on the torchtitan engine that is shorter than
torchtitan's default interval, or whose `save_freq` is not a multiple of it,
currently produces no usable checkpoint. That is every short SFT smoke run and
most debug-scale RL runs.

### Test plan

Run the torchtitan-engine SFT recipe with `trainer.save_freq=1` and
`trainer.total_epochs` short enough that the run ends before torchtitan's
default interval, then assert the checkpoint directory contains model shards
(`__*.distcp` with model FQNs), not just dataloader state. Before the change
the model keys are absent; after, they are present.

---

## Notes for the filer

- The fork commit carries a `[kimi_k3]` prefix, but **nothing about this bug is
  kimi-specific** — drop the prefix when filing; the failure is in the shared
  engine and hits every model.
- Keep this PR separate from the CP fixes (PR18). Different bug, different
  reviewer surface, and this one is a one-liner that can land immediately.
