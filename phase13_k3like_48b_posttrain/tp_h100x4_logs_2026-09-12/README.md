# TP/SP #4499 on 4 x H100 PCIe (2026-09-12)

Branch `tp_sp_on_main` = `22eeec412` (on upstream main `7e7f271e0`), reference main `7e7f271e0`. Box: 4 x NVIDIA H100 PCIe 80 GB (capability 9.0; NVLink between GPUs 0-1 and 2-3, PCIe across), driver 580.159.04, torch `2.15.0.dev20260906+cu130`, spmd_types 0.2.5, torch_remat 0.2.0, Attention Gym main `499404b` on `PYTHONPATH`, KDA guard widened locally in both trees. Kit: `matrix_scripts/tp_h100_v2/` (`run_v2.sh`, `run_v2_dp2.sh`, `run_v2_k27.sh`).

- `tp_v2_run.log`: dp1, 256 tokens per step, 100 steps. tp=1 on the branch is identical to main on all 100 steps (loss and grad norm), twice. tp=2 SP on / off: +0.051 / +0.027 % at step 1, -5.59 / -6.76 % at step 10, -1.83 / +4.28 % at step 100. The numbers equal the 2026-09-11 2 x H100 PCIe run on the pre-rebase head, value for value; the 2026-09-12 2 x H100 SXM run (`tp_h100_logs_2026-09-12/`) differs from step 2 on, the box's effect.
- `tp_v2_dp2.log`: dp2, 512 tokens per step (256 per rank; the multimodal collator needs one whole row per rank), 100 steps, reference dp2 on the branch. dp2 x ep2 (no TP) -10.7 % at step 10; dp2 x tp2 -6.0 %, dp2 x ep2 x tp2 -6.9 %. The reference is memorised by step 100 (0.78).
- `tc_tp2.log`, `tc_dp2_ep2_tp2.log`: 3 steps with `--debug.spmd_typechecking` and activation checkpointing off; complete, the only "error" lines are the missing `lspci` warning.
- `k27_*.log`: Kimi K2.5 dp2 tp=1, branch against main, no seed checkpoint (K2.5 cannot run on one GPU on main).
