# PR #13 — filing instructions

## Status

🟠 **Issue body ready; PR patch ready in spec but needs local validation run before filing.**

## Where to file

| Repo | URL | What to file |
|---|---|---|
| fla-org/flash-linear-attention | https://github.com/fla-org/flash-linear-attention/issues/new | First the issue body (cross-link to closed PR #796) |
| same | Then a draft PR against `main` | Reference the issue, target `fla/modules/fused_norm_gate.py` only |

## Title

```
[Bug] layer_norm_gated_{fwd,bwd}_kernel: same NB-autotune-key + BS<BT crash as #796 layernorm fix, but in fused_norm_gate.py — Blackwell sm_120 + Triton 3.6.0
```

## Body

Use [PR.md](PR.md) → top section ("Background" through "Why this matters") verbatim as the issue body. The "Proposed fix" diff goes in the PR description, and the "Tests to add" goes in a follow-up comment after CI confirms green.

## Pre-flight checks before filing

- [ ] Apply the proposed diff to a local fla checkout (don't hot-patch site-packages — auto-classifier rejects that).
- [ ] Run `pytest tests/modules/test_fused_norm_gate.py` against the patched checkout — confirm baseline passes.
- [ ] Add the mirrored regression tests (`test_rmsnorm_gated_varying_nb_{no,with}_residual`); confirm they fail without the diff and pass with.
- [ ] Re-run our phase 5 stage 2 SFT for ≥ 8 000 steps (>= 3× the empirical crash window) and confirm no `device-side assert triggered` in any stage2_attempt*.log.
- [ ] Note any int64 overflow concerns (PR #818 covers the broader pointer-arithmetic class) — may or may not bundle.

## Related upstream activity

- ✅ PR #795 (merged 2026-02-14): `Fix layer_norm_bwd_kernel OOB access on high-SM GPUs` — fixed the same class of bug in `layernorm.py` bwd kernel for idle programs.
- 🔵 PR #796 (open): `[Layernorm] Fix autotuner crash and OOB writes in layer_norm_bwd on high-SM GPUs` — fixes the residual two bugs (NB key + BS<BT) in `layernorm.py`.
- 🔵 PR #818 (open): `[Ops] Fix int32 overflow in pointer arithmetic across all Triton kernels` — applies to ~83 files. Our case (T=4640) doesn't trigger int32 overflow, but the patch may touch `fused_norm_gate.py` too. Check before filing to avoid conflict.
- ❌ No open issue for `fused_norm_gate.py` specifically — this PR is the first to call it out.

## Our local workaround (not in repo)

The patch in PR.md applied to `/usr/local/lib/python3.12/dist-packages/fla/modules/fused_norm_gate.py` is the minimum hot-fix to unblock stage 2 SFT runs. Auto-classifier blocks the direct edit; the user can manually apply if they want stage 2 to stop crashing while the upstream PR is in review.

Long-term: vendor fla into the torchtitan submodule or pin a forked fla branch with the patch.
