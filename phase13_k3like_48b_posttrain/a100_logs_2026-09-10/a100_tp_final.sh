#!/bin/bash
# The TP/SP table for PR 4499 in PR 4500 format: the reworked head (9a62f5229 + the local Ampere gate commit) against its parent main ac10ca48f, bf16, seed 42, deterministic, one seed checkpoint per stream, 100 steps; spmd_types cells (tensor parallelism requires it), tp=1 on both backends as the control. Then QB dp8 x ep8 for 100 steps.
source /workspace/venv/bin/activate
export PYTHONPATH=/workspace/attn_gym_up TT=/workspace/wt_tpmain TT_PARENT=/workspace/wt_mainparent OUT=/workspace/tp_final_out
mkdir -p $OUT; cd /workspace/matrix_scripts/tp_a100 && . ./common.sh
stamp() { echo "=== $1 $(date +%T)"; }
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
B2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
stamp "seeds"; seed bf16 $B; seed bf16_dp2 $B2
stamp "dp1 stream wave 1"
cell tp1_parent    $TT_PARENT 0   1 bf16 100 $B $D 1 $PD &
cell tp1_parent_st $TT_PARENT 1   1 bf16 100 $B $D 1 $ST &
cell tp1           $TT        2   1 bf16 100 $B $D 1 $PD &
cell tp1_st        $TT        3   1 bf16 100 $B $D 1 $ST &
cell tp2_sp_st     $TT        4,5 2 bf16 100 $B $D 1 $T 2 $ST &
cell tp2_nosp_st   $TT        6,7 2 bf16 100 $B $D 1 $T 2 $ST $NOSP &
wait
stamp "dp1 stream wave 2"
cell tp4_sp_st     $TT 0,1,2,3 4 bf16 100 $B $D 1 $T 4 $ST &
cell tp4_nosp_st   $TT 4,5,6,7 4 bf16 100 $B $D 1 $T 4 $ST $NOSP &
wait
echo "# dp1 stream (reference: tp=1 on the parent):"; table tp1_parent tp1_parent_st tp1 tp1_st tp2_sp_st tp2_nosp_st tp4_sp_st tp4_nosp_st
stamp "dp2 stream"
cell dp2_st        $TT 0,1     2 bf16_dp2 100 $B2 $D 2 $ST &
cell dp2_tp2_st    $TT 2,3,4,5 4 bf16_dp2 100 $B2 $D 2 $T 2 $ST &
cell dp2_ep2_st    $TT 6,7     2 bf16_dp2 100 $B2 $D 2 $E 2 $ST &
wait
cell dp2_ep2_tp2_st $TT 0,1,2,3 4 bf16_dp2 100 $B2 $D 2 $E 2 $T 2 $ST &
cell dp2_parent_st  $TT_PARENT 4,5 2 bf16_dp2 100 $B2 $D 2 $ST &
wait
echo "# dp2 stream (reference: dp2 on this branch; dp2 on the parent as the control):"; table dp2_st dp2_parent_st dp2_tp2_st dp2_ep2_st dp2_ep2_tp2_st
stamp "qb ep 100"
rm -rf /workspace/mx3_qb100_*
timeout 7200 bash /workspace/matrix_scripts/qb_dp8ep8_a100.sh 2>&1 | tail -6
stamp "done"; echo TP-FINAL-DONE
