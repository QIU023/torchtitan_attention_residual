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

## Status at 11:00Z

### veRL multimodal across the parallelism axes: seven cells green, engine unchanged

    cell             shape                  steps  logprob diff  prompt length
    img_tp2          tp 2                   3      0.14548       111.0
    img_cp2          cp 2                   3      0.13497       111.0
    img_ep2          fsdp 2 x ep 2          3      0.13499       111.0
    img_pp2          pp 2                   3      0.11449       111.0
    img_cp2tp2       cp 2 x tp 2            3      0.14940       111.0
    img_pp2cp2       pp 2 x cp 2            3      0.10457       111.0
    img_fsdp2pp2ep2  fsdp 2 x pp 2 x ep 2   3      0.11079       111.0

Every cell reads the same prompt length, 111, which is the expanded media block the builder produces offline for one row of this parquet, so each one carried its images into the policy instead of dropping them. The log-prob diffs stay in the class the text cells read on this debug model, and the gradient norms in 3.80 to 4.11. No engine change was needed for any of them, which is what the code read of the three vision routes predicted.

Two findings about the harness rather than the engine: the cell script ends with an echo so its exit code is always zero and the step count is the only verdict; and `img_tp2ep2` as first written asked for four GPUs with `dp_shard 1, tp 2`, which torchtitan refuses because expert parallelism divides `dp_shard * cp * tp` instead of being a world dimension. That row is void and the cell is being rerun on two GPUs.

### Also done since 10:30Z

- The gradient probe and the gradient comparison both proven on this box: two ranks on the standard dispatcher from the cached seed, `GRAD_PROBE_OK loss=7.762686 n_params=726 total_norm=25.40821`, and the comparison resolving all 726 parameters into its eight groups with zero difference against itself. The `num_valid_tokens` fix holds, so the next H200 session starts from working scripts.
- The matrix runner's error filter fixed (it was matching the trainer's own configuration dump) and the table generator rewritten to drop the meaningless exit code and pick up hand-relaunched cells.

### Running now

The relaunched `img_tp2ep2` on two GPUs, and the drop-images comparison on two more: the same batch twice, once normally and once with every vision tensor dropped before the forward, which is the direct evidence that the tower reaches the policy.

### Next, when those two free the GPUs, in this order

1. **The two eight-GPU cells of the image matrix, which have not run yet** (`CHAIN=C bash verl_img5d.sh`): `img_dp2cp2tp2ep2` (fsdp 2 x cp 2 x tp 2, with ep 2 dividing that product) and `img_pp2tp2cp2` (pp 2 x tp 2 x cp 2). They are the widest shapes this box can hold and they close the axis coverage: every pair of axes and, between the two of them, all five at once across the matrix. Eight GPUs is the physical ceiling here, since `dp x cp x tp x pp` must equal the device count and expert parallelism divides `dp x cp x tp`; five axes simultaneously would need sixteen.
2. The 21-cell integration matrix on the rebased tree (`run_mx4_int0917.sh`), step-1 of every cell against `mx4_int0916c`.

The user's order was veRL multimodal first, then the integration tree, so chain C goes ahead of the matrix.

## Status at 10:30Z

Done and pushed:

- veRL fork rebased onto upstream verl main `1a8a0f5f` (64 commits, no conflict, per-file diffs unchanged, 26 CPU tests pass), then the pipeline token budget factored into `pipeline_token_budget()` with five CPU tests. Head `6f71c346`.
- The split kit regenerated and verified on that head (`split_2026-09-17/`, base `1a8a0f5f`): the union reproduces the head, every stage parses and is ruff-clean, no local marker in patches 01 to 06.
- The integration tree rebased onto torchtitan main `a3a819c67` (81 commits, one conflict in `config/transform/context_parallel.py` resolved to the tree's loop with main's typing), head `31a1b39fa` in `/tmp/wt_int0917`; 153 CPU tests pass, the one failure and the five uncollectable files are this box's missing packages and fail the same way on pristine main. Launcher `run_mx4_int0917.sh` ready, every flavor it names verified present.
- MoonEP: body requirements, limitations and the H200 test plan; the next-session order (`MOONEP_NEXT_H200_2026-09-17.md`); the hot probe's standard-path branch wrapped in the ambient mesh; the gradient probe's token-count fix. RFC 3029's MoonEP, AC and 5D rows refreshed.
- The image path read against TP, PP and CP before running anything, and the first two cells (tp 2, cp 2) pass with images actually carried.

Running: the image cells across the axes, two chains. Waiting on GPUs: the 21-cell integration matrix, the drop-images diagnostic, the MoonEP gradient probe.
