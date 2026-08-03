# PR #19 — common MoE sharding: EP-off drops the computed `in_grad_placements` (silent gradient under-reduction, reproduces on deepseek_v3)

**Target**: `pytorch/torchtitan`, `torchtitan/models/common/moe_sharding.py` (`set_moe_sharding_config`)
**Fork reference**: `attention_residual_dev` @ `0d83910c4` — **`moe_sharding.py` hunk only**; the kimi_k3 hunk in that commit reverts a fork-internal workaround and must not be included.
**Upstream audit**: 2026-08-03 @ `681fd4b50` — `if enable_ep else None` still at L311-322, patch applies cleanly. **Parallel-safe with PR16** (different files, no overlap).
**Risk**: low — with EP the passed tuple is identical to today's; only EP-off changes, from `None` to the layout the function already computes.

---

## Suggested PR title

> [MoE] fix: pass the computed `in_grad_placements` without EP too — TP gradients below the experts lose their reduction

## Suggested PR body

### Summary

`set_moe_sharding_config` computes `experts_in_grad_layout` for both branches
(`dense_activation_placement(tp=spmd.P)` when EP is off) but passes it to
`LocalMapConfig` only `if enable_ep else None`. The EP-off value is computed
and thrown away, and `local_map` defaults the input gradient placements to the
input placements — `Replicate`.

Without EP the experts run replicated on the tp axis, so each rank's gradient
w.r.t. the local_map inputs is one contribution to a sum — exactly what the
discarded `tp=spmd.P` layout declares. Defaulting to `Replicate` skips the
all-reduce and keeps one rank's share. The forward is untouched, so loss
curves look normal; every parameter the backward reaches below the experts
(router gate included) gets an under-reduced gradient. Only TP-without-EP
topologies are affected.

### Evidence — unmodified `deepseek_v3_debugmodel`

Seeded run (shared step-0 checkpoint), dp1, varying only tp. Ratio =
per-parameter materialized grad-norm dp1/tpN. Before the fix, the five router
gates are the model's five worst parameters:

|                              | before tp2 | after tp2 | after tp4 |
|---|---|---|---|
| `layers.2.moe.router.gate`   | 1.4780 | 1.0011 | 0.9999 |
| `layers.4.moe.router.gate`   | 1.4401 | 0.9998 | 0.9974 |
| `layers.3.moe.router.gate`   | 1.4178 | 1.0003 | 1.0009 |
| `layers.5.moe.router.gate`   | 1.2109 | 1.0022 | 1.0009 |
| `layers.1.moe.router.gate`   | 1.2067 | 0.9975 | 0.9992 |
| max \|ratio−1\| (83 params)  | 0.4780 | 0.0025 | 0.0026 |

The pre-fix deviations sit near `sqrt(tp)` — a dropped sum of near-orthogonal
per-rank shares, not a wrong scale factor. Post-fix grad_norm is
3.2649 / 3.2650 / 3.2651 across tp 1/2/4. The model and parallelism code
under test are entirely upstream torchtitan; our only addition is the probe
harness that reads per-parameter materialized gradients.

### Fix

Pass the tuple unconditionally (EP branch byte-identical to today):

```python
in_grad_placements=(
    experts_in_grad_layout,
    experts_in_grad_layout,
    experts_in_grad_layout,
    _tokens_per_expert_placement(enable_ep=enable_ep),
),
```

### Test plan

Reproduction recipe above. Happy to add a DTensor unit test pinning
`local_map`'s input-gradient placements under a tp mesh with
`enable_ep=False` (fails before, passes after). Existing tests unaffected.

---

## Notes for the filer

- **No mandatory GPU gate** — unlike PR16, the before/after evidence above was
  already measured on the box (2026-07-31 session, `tp_trainer_grad_probe.py`,
  fork checkout whose `moe_sharding.py` region is verified identical to
  upstream `681fd4b50`). Two optional upgrades if the PR16 gate worktree is
  already up: (a) re-run the before/after probe against pristine upstream
  `681fd4b50` (vanilla vs +patch, dp1/tp2, <=2 GPUs) so the table is
  literally-upstream; (b) write the DTensor unit test promised in the test
  plan (fails before / passes after) so this PR ships a test like PR16 does.
  Neither blocks filing.
- Lead with the deepseek_v3 reproduction — it needs none of our model code.
- Provenance, one sentence max: found while root-causing a TP gradient gap on
  our Kimi-K3-family model (public logbook, TP_GRAD_FINDING_2026-07-29.md).
- If asked why the fix belongs in the shared config and not per-model: a
  model-side re-reduction was tried first and reverted — stacked with this
  fix, the same edge reduced twice per layer and 21-layer grad_norms exploded
  to ~1e7.
- Can be opened the same day as PR16 — cross-link once numbers exist.
