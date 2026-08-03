# PR #19 — common MoE sharding: EP-off drops the computed `in_grad_placements`, silently under-reducing gradients below the experts (reproduces on deepseek_v3)

**Target repo**: `pytorch/torchtitan`
**Target file**: `torchtitan/models/common/moe_sharding.py` (`set_moe_sharding_config`, the `LocalMapConfig` construction)
**Fork reference**: torchtitan `attention_residual_dev`, commit `0d83910c4` (the `moe_sharding.py` hunk only; the same commit also reverts a model-side workaround that is not part of this patch)
**Upstream audit (2026-08-03)**: `upstream/main` @ `681fd4b50` still has `in_grad_placements=( (...) if enable_ep else None )` at L311-322. Patch applies cleanly. **Not obsoleted; still applies.**
**Effort**: ~half a day (patch is landed and measured; a DTensor unit test is the main add).
**Risk**: low — with EP enabled the passed tuple is identical to today's; only the EP-off branch changes, from `None` (defaulted placements) to the layout the function already computes.

---

## Suggested PR title

> [MoE] fix: pass the computed `in_grad_placements` without EP too — TP gradients below the experts lose their reduction

## Suggested PR body

### Summary

`set_moe_sharding_config` computes `experts_in_grad_layout` for BOTH branches —
`dense_sequence_parallel_placement()` with EP, and
`dense_activation_placement(tp=spmd.P)` without — but then passes the tuple to
`LocalMapConfig` only `if enable_ep else None`:

```python
in_grad_placements=(
    (
        experts_in_grad_layout,
        experts_in_grad_layout,
        experts_in_grad_layout,
        _tokens_per_expert_placement(enable_ep=enable_ep),
    )
    if enable_ep
    else None
),
```

So the EP-off value is calculated and thrown away, and `local_map` defaults the
input gradient placements to the input placements — `Replicate`.

Without EP the experts run replicated on the tp axis, so each rank's gradient
w.r.t. the local_map inputs is **one contribution to a sum**, not the finished
value — exactly what the discarded `tp=spmd.P` layout declares. Defaulting to
`Replicate` tells DTensor the gradient is already consistent, skips the
all-reduce, and keeps one rank's share.

**The forward is untouched and correct**, which is why loss curves never showed
it. Every parameter the backward reaches after the experts (router gate, and in
latent-MoE designs everything upstream of the expert input) receives an
under-reduced gradient.

### Evidence — unmodified `deepseek_v3_debugmodel`

Seeded run (shared step-0 checkpoint so all arms start from identical weights),
dp1, varying only tp; ratio = per-parameter materialized grad-norm dp1/tpN.
Before the fix, the model's five router gates are its five WORST parameters:

|                              | before tp2 | after tp2 | after tp4 |
|---|---|---|---|
| `layers.2.moe.router.gate`   | 1.4780 | 1.0011 | 0.9999 |
| `layers.4.moe.router.gate`   | 1.4401 | 0.9998 | 0.9974 |
| `layers.3.moe.router.gate`   | 1.4178 | 1.0003 | 1.0009 |
| `layers.5.moe.router.gate`   | 1.2109 | 1.0022 | 1.0009 |
| `layers.1.moe.router.gate`   | 1.2067 | 0.9975 | 0.9992 |
| max \|ratio−1\| (83 params)  | 0.4780 | 0.0025 | 0.0026 |

The pre-fix deviations sit near `sqrt(tp)` — the signature of a sum of
near-orthogonal per-rank shares where only one share survives (a dropped
reduction), not of a wrong scale factor. After the fix, grad_norm at step 1 is
3.2649 / 3.2650 / 3.2651 across tp 1/2/4.

Nothing in this reproduction uses any code outside upstream torchtitan.

### Why it went unnoticed

- The forward is exact; only backward placements are wrong.
- The deviation is per-parameter and bounded (~sqrt(tp) on the router gate), so
  training still converges plausibly; grad clipping and any gradient-norm-based
  diagnostics are what actually drift.
- With EP enabled (the config most MoE runs use at scale) the declaration IS
  passed, so the bug only lives in TP-without-EP topologies.

### Fix

Pass the tuple unconditionally — the EP branch is byte-identical to today's
behaviour, the EP-off branch now uses the layout the function already computed:

```python
in_grad_placements=(
    experts_in_grad_layout,
    experts_in_grad_layout,
    experts_in_grad_layout,
    # num_local_tokens_per_expert_E is routing metadata, but it is
    # still a DTensor input to local_map and must have placements.
    _tokens_per_expert_placement(enable_ep=enable_ep),
),
```

### Test plan

- Reproduction recipe above (seeded `deepseek_v3_debugmodel`, dp1 vs tp2/tp4,
  per-parameter materialized gradients via `full_tensor()`); happy to
  contribute the probe as a unit/integration test if wanted — a DTensor-level
  unit test would pin `local_map`'s input-gradient placements under a tp mesh
  with `enable_ep=False`, asserting the gradient w.r.t. the experts' input
  all-reduces over tp (fails before, passes after).
- Existing tests: unaffected — the EP-enabled path passes the identical tuple.

---

## Notes for the filer

- **Independent of PR16** (`moe.py` routing-map scatter): different file,
  different defect class (forward placement preservation vs backward gradient
  declaration), no textual overlap — the two patches were verified to apply
  cleanly to the same upstream main (`681fd4b50`) in either order. **They can
  be opened in parallel**; cross-link the numbers in both bodies once filed
  ("related but independent MoE-under-TP fix").
- Lead with the deepseek_v3 reproduction — it needs none of our code, which
  makes this the cleanest core fix we have.
- This fix was found while root-causing a TP gradient gap on our Kimi-K3-like
  model (public logbook: TP_GRAD_FINDING_2026-07-29.md in
  QIU023/torchtitan_attention_residual). Mention it once as provenance; keep
  the body upstream-centric.
- Do NOT bundle the kimi_k3-side `_NoParallelPartialGrad` revert from
  `0d83910c4` — that half is fork-internal (it removed a workaround this fix
  supersedes; with both in place the same gradient edge reduced twice and
  grad_norms exploded ~1e7 at 21 layers — worth one sentence in review if
  asked why the fix must live in the shared config and not per-model).
