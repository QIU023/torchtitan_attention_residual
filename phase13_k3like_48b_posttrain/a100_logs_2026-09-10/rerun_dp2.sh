#!/bin/bash
# The bf16 dp2 stream again with the 512-token train step, after the main driver finishes.
until grep -q ALL-DONE /workspace/a100_run_bf16first.log 2>/dev/null; do sleep 60; done
source /workspace/venv/bin/activate
export PYTHONPATH=/workspace/attn_gym_up TT=/workspace/wt_tp TT_PARENT=/workspace/wt_parent OUT=/workspace/tp_a100_out
cd /workspace/matrix_scripts/tp_a100 && . ./common.sh
B2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
seed bf16_dp2 $B2
cell dp2         $TT 0,1     2 bf16_dp2 100 $B2 $D 2 $PD &
cell dp2_tp2     $TT 2,3,4,5 4 bf16_dp2 100 $B2 $D 2 $T 2 $PD &
cell dp2_ep2     $TT 6,7     2 bf16_dp2 100 $B2 $D 2 $E 2 $PD &
wait
cell dp2_ep2_tp2 $TT 0,1,2,3 4 bf16_dp2 100 $B2 $D 2 $E 2 $T 2 $PD &
cell dp2_tp2_st  $TT 4,5,6,7 4 bf16_dp2 100 $B2 $D 2 $T 2 $ST &
wait
table dp2 dp2_tp2 dp2_tp2_st dp2_ep2 dp2_ep2_tp2
echo DP2-RERUN-DONE
