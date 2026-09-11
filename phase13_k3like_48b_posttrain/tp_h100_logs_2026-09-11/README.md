# TP/SP matrix for PR 4499 on 2 x H100 PCIe (2026-09-11)

Kit: `matrix_scripts/tp_h100/run_bf16_100.sh` run verbatim. Branch `tp_sp_on_main` = `bd55160a8`
(five commits on main `da2f82670`; the morning's review head `d4d6e774c` plus one commit, the
dispatcher condition dropping the tensor-parallel disjunct). Parent worktree `da2f82670`. The only
local change on either tree is the kit's `kda_capability_hack.py` (SM 9.0 admitted), uncommitted.
torch `2.15.0.dev20260906+cu130`, Attention Gym upstream main `b16d6d3`, capability 9.0.
Protocol: bf16, 256 tokens per step (512 for the dp2 stream), 100 steps, one seed checkpoint per
batch shape built on the branch tree, one inductor cache per cell. `tables.txt` is the kit's output;
one log per cell.

Not run: the four-card cells (dp2 x tp2, dp2 x ep2 x tp2, tp4), the box has two GPUs.

Not comparable with the A100 table in the PR body: that was the previous stack (`9a62f5229`), with
the vision tower replicated on TP; this stack shards the tower.

Outlier, not a regression. The first `tp1_pd` run (branch, partial_dtensor, tp=1) read bitwise
with main at step 1 and then diverged (-2.96% at step 10, -6.6% at step 20). It did not reproduce:
rerun on the same seed checkpoint with a fresh inductor cache (`tp1_pd_again`), the branch is bitwise
with main at every step, and so is the parent's rerun (`tp1_parent_pd_again`); the first run differs
from its own rerun by +3.05% at step 10 and +7.06% at step 20. The non-TP path is therefore unchanged
by the branch, as the kit expects. Cold inductor compiles picking a different kernel have produced
this pattern on these boxes before; that cause is not established for this cell. Both comparisons are
in `rerun_table.txt`.
