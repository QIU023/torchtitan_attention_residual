# PR #16 — RETIRED 2026-08-03: the bug is torch-2.12-only; do not file

**Status**: **DO NOT FILE.** The defect this kit packaged does not exist on
any torch upstream targets. The fork keeps the fix as a torch-2.12-stable
compat shim (recategorized from "core fix" to "compat shim" in the
core-changes audit); drop it when the fleet moves to >=2.13.

## Evidence chain that retired it

1. **torch 2.12.0 stable** (original environment): bare in-place `scatter_`
   on `Shard(1)` router outputs errors — live crash on the K3 model under
   TP+EP, and the kit's unit test FAILED against the bare construction.
2. **torch 2.13.0 stable** (`scatter_probe.py` in this folder, 2026-08-03,
   CPU gloo, 2 ranks — the identical construction the unit test uses:
   `distribute_tensor([Shard(1)])` scores/topk, bare
   `zeros_like(...).scatter_(-1, topk, True)`):

   ```text
   [rank0] torch=2.13.0+cpu SCATTER OK placements_kept=True value_correct=True local_counts=[4, 4, 2, 3]
   [rank1] torch=2.13.0+cpu SCATTER OK placements_kept=True value_correct=True local_counts=[4, 2, 4, 4]
   ```

   Placement preserved, values exact, counts per-shard — the full
   `Partial(sum)` contract, with no fix. torch 2.13 added the missing
   `aten.scatter_` sharding strategy.
3. **torch 2.14 nightly** (GPU box, pristine upstream `681fd4b50`,
   `deepseek_v3_debugmodel` dp_shard2 x tp2 x ep2): trains clean under BOTH
   `spmd_backend default` and `full_dtensor` (loss 8.01456 / 6.19790).
   Consistent with (2).

## Corrections to earlier revisions of this kit

- The "declarative backend vs default backend" trigger theory (2026-08-03
  interim revision) was wrong: deepseek_v3's parallelize calls
  `model.parallelize(parallel_dims)` under tp/ep on the DEFAULT backend too
  (parallelize.py L49-50) — the declarative MoE sharding configs are active
  either way. The only discriminating variable across every observation was
  the torch version.
- The `moe.py` comment this kit called stale ("DTensor runs it as a local
  op") is CORRECT on torch >= 2.13. It was wrong only on 2.12.
- The unit test's "fails before the fix" property holds only on 2.12; on
  >= 2.13 the bare construction passes it.

## What survives

- The fork's fix (`129e29de`) is harmless on every version (its DTensor
  branch reproduces exactly what >= 2.13 does natively) and stays as the
  2.12 compat shim. It must still be EXCLUDED from the K3 upstream PR
  branch — same handling as the other stable-torch shims.
- `scatter_probe.py` — the 40-line version-discriminating instrument; rerun
  it on any torch to re-check.
- Method note for the logbook: an op-coverage bug in a moving dependency
  needs a version-pinned minimal probe BEFORE it becomes a PR claim. Two
  full reproduction attempts on the box were spent discovering what the
  probe shows in seconds.

## PR19 is unaffected

PR19's defect (EP-off `in_grad_placements` computed then dropped) is a
sharding-config declaration bug, not a DTensor op-coverage gap: it is in the
current upstream source (audited @ `681fd4b50`), and the DSv3 reproduction
recipe is valid on current upstream with no backend flag — DSv3 consumes the
MoE sharding configs (and thus `LocalMapConfig`) under plain tp on the
default backend.
