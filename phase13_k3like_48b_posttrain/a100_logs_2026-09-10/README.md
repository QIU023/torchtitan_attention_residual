# A100 box archive (2026-09-10)

Everything small from `/workspace` on the 8 x A100-SXM4-40GB box (`137.175.22.196:25133`) pulled back before the instance was released: driver scripts and logs, the matrix kit (`matrix_scripts/`, with the local probe patches), and the per-cell measure logs and `results.txt` of every run. Checkpoints, the inductor caches, the venv and the source worktrees are not here; the local commits on those worktrees ("A100 local: ... (not for upstream)") are the eager reference gate below SM90 and the KDA capability guard at >= 8.0, both one-liners recorded in `TP_SP_ON_4500_2026-09-09.md`.

- `a100_tp_final.log`, `tp_final_out/`: the final TP/SP tables for PR 4499 (head `9a62f5229` vs main `ac10ca48f`, dp1 and dp2 streams).
- `tp_a100_out/`: the earlier 4527-based table (superseded).
- `out_4500/`, `run_4500_cp.log`: the PR 4500 replication (all-gather and Ulysses CP).
- `mx3_qb100_dp8ep8_ctrl_0910_090628/`, `a100_qb_final.log`, `rerun_qb.log`: the QB dp8 x ep8 stage; stopped before it produced numbers (see the note in `TP_SP_ON_4500_2026-09-09.md`).
