# Overnight plan, 2026-09-17 (5060 box, no H200)

State at logoff: MoonEP PR 4751 head `84f2704ce` (fused transport + lint + the on-device GPU test); AC reuse PR 4656 head `7e9622a22` with the H200 tables in its body; H200 paused with the MoonEP chain partly done (see `MOONEP_H200_2026-09-17.md`).

## torchtitan main line

1. MoonEP numerics, the open item: `dp2ep2_moonep` and `dp4ep4_moonep` differ from the standard dispatcher at step 1 (12.52362 vs 12.52567 at dp2; 12.56253 vs 12.54318 at dp4) while the fresh-cache floor rows are bitwise, so the difference is the transport's own rounding, not the cache. It needs NVSwitch to measure, so tonight only the analysis: finish the standard-path branch of `moonep_hot_probe.py` (the core dispatcher wants `set_current_mesh(mesh)` around its calls) so the next H200 session prints std-vs-fp32, moonep-vs-fp32 and std-vs-moonep on one batch, and fix `grad_probe.py` (its rc=1 on the fused head is unread: `logs_moonep_2026-09-17_h200/results/gp_std_np2.log`). Nothing of the moonep rows goes into the body until that is done.
2. Integration tree `k3_on_4025` (`cf7418637` on main `810e62786`): rebase onto `a3a819c67` (five commits newer, renderers dependency), rerun the 21-cell seeded matrix on this box, keep the step-1 identities.
3. PR bodies: 4312 follow-up comment (`COMMENT_4312_FOLLOWUP_2026-09-17.md`, the user posts); PP offload (ready per the CPU box) and PP balance (not) per `PR_BODY_PP_OFFLOAD.md` / `PR_BODY_PP_BALANCE.md`; the README parallelisms PR kit.
4. RFC 3029 status table: refresh the MoonEP and AC rows with the H200 facts.

## veRL side (branch `kimi_k3_integration_rebased` = `549c1e21`)

1. The with/without-images step-1 log-prob check of the image GRPO cell (direct evidence of the actor's vision path), then the same cell under TP (the `spmd_types` annotation of vision tensors outside CP).
2. Rebase the fork onto upstream verl main, regenerate the split patches (`Raising_PRs/PR_VERL_K3/split_2026-09-17/work/build.py`), rerun the CPU/gloo tests.
3. When torchtitan PR 4760 merges: add `MLAFlexInnerAttention.Config` to the engine's CP transform mapping (patch 04).
4. No Qwen or upstream-model cells unless asked; at most two Ray clusters.
