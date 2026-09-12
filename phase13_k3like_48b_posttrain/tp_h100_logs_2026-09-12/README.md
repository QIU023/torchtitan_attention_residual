# TP/SP #4499 on 2 x H100 80GB HBM3 (2026-09-12)

Branch `tp_sp_on_main` = `22eeec412` (four commits on upstream main `7e7f271e0`, #4419, which removed the `spmd_backend` option and the partial_dtensor backend). Reference: tp=1 on main `7e7f271e0`, a second worktree. Box: 2 x NVIDIA H100 80GB HBM3 (SXM, NVLink NV6, capability 9.0), driver 580.173.02, torch `2.15.0.dev20260906+cu130`, spmd_types 0.2.5, torch_remat 0.2.0, Attention Gym upstream main `499404b` on `PYTHONPATH`; the KDA capability guard widened locally in both trees (`tp_h100_v2/hacks/kda_capability_hack.py`), not part of the PR.

Protocol: #4500's -- dp1, `seed=42`, `--debug.deterministic`, bf16, 256 tokens per step as one 256-token micro-batch, one seed checkpoint, one inductor cache and a per-rank Triton cache per cell, 100 steps; table at steps 1 / 10 / 100. Kit: `matrix_scripts/tp_h100_v2/run_v2_2gpu.sh`.

Results (`tp_v2_run.log`): tp=1 on the branch is identical to main on all 100 steps, loss and grad norm, and again on a fresh inductor cache. tp=2 with SP: +0.051 / +0.80 / +0.68 % loss at steps 1 / 10 / 100; tp=2 without SP: +0.027 / +2.54 / -3.59 %. #4500's own CP=2 rows read +3.8 % and +15.1 % at step 10 on H100.

`tc_tp2`: 3 steps at tp=2 with SP, `--debug.spmd_typechecking`, activation checkpointing off (type checking refuses selective AC with flex attention; the recipes' `_set_spmd_typechecking` does the same). No traceback; the log carries no AC line, while the tabulated cells apply SelectiveAC. The four-GPU smoke (dp2 x ep2 x tp2 with type checking) did not run: this box has two GPUs.
