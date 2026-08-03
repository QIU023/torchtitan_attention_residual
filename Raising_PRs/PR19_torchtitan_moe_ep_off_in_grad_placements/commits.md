# Source commits (fork torchtitan `attention_residual_dev`)

| commit | scope |
|---|---|
| `0d83910c4` | `moe: declare in_grad_placements without EP too -- the layout was computed then dropped` — 2 files. **Only the `torchtitan/models/common/moe_sharding.py` hunk (+24/−16 incl. comment) goes upstream**; the `torchtitan/experiments/kimi_k3/parallelize.py` hunk reverts a fork-internal workaround (`_NoParallelPartialGrad`) that this fix supersedes and must NOT be included. |

Files here:

- `moe_sharding_in_grad_placements_PR19.patch` — the upstream half of the fix,
  extracted as `git diff 0d83910c4~1 0d83910c4 -- torchtitan/models/common/moe_sharding.py`.
  Verified 2026-08-03 to apply cleanly to `pytorch/torchtitan` main @ `681fd4b50`
  (checked in a clean worktree with `git apply --check`).

Evidence chain (public logbook, QIU023/torchtitan_attention_residual):

- `phase13_k3like_48b_posttrain/TP_GRAD_FINDING_2026-07-29.md` — the full
  root-cause narrative ("2026-07-31: the MoE defect was two bugs" section:
  tensor-level probe showing tp2 rank shares 0.02921830 + 0.02839850 summing to
  the tp1 value 0.04075606, i.e. the gradient is genuinely Partial).
- `phase13_k3like_48b_posttrain/tp_trainer_grad_probe.py` — the seeded
  per-parameter instrument used for the deepseek_v3 reproduction table.

Upstream state verified 2026-08-03 against `pytorch/torchtitan` main
@ `681fd4b50`: `set_moe_sharding_config` still passes
`in_grad_placements=( (...) if enable_ep else None )` (moe_sharding.py L311-322).

Parallel-filing note: no shared files with PR16 (`moe.py`); both patches
apply cleanly to `681fd4b50` independently and in either order.
