#!/bin/bash
# TP/SP #4499 on tp_sp_on_main (22eeec412, on main 7e7f271e0), #4500's format: dp1, bf16, 256 tokens
# per step, 100 steps, table at steps 1 / 10 / 100, reference tp=1 on main. Four GPUs, two waves.
# Then two type-checking smokes (3 steps, not tabulated). Set TT (branch tree), TT_PARENT (main tree), OUT.
set -u; . "$(dirname "$0")/common.sh"
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
B2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
seed bf16 $B
seed bf16_dp2 $B2
# wave 1
cell tp1_parent $TT_PARENT 0   1 bf16 100 $B $D 1 &
cell tp1        $TT        1   1 bf16 100 $B $D 1 &
cell tp2_sp     $TT        2,3 2 bf16 100 $B $D 1 $T 2 &
wait
# wave 2
cell tp2_nosp   $TT        0,1 2 bf16 100 $B $D 1 $T 2 $NOSP &
cell tp1_again  $TT        2   1 bf16 100 $B $D 1 &
wait
# type-checking smokes: the declarations, not numerics
TAIL=activation_checkpoint:none cell tc_tp2        $TT 0,1     2 bf16     3 $B  $D 1 $T 2 $TC &
wait
TAIL=activation_checkpoint:none cell tc_dp2_ep2_tp2 $TT 0,1,2,3 4 bf16_dp2 3 $B2 $D 2 $E 2 $T 2 $TC
echo
echo '# dp1, 256 tokens per step (reference: tp=1 on main; tp1 and tp1_again must be bitwise with it)'
table tp1_parent tp1 tp1_again tp2_sp tp2_nosp
echo
for c in tc_tp2 tc_dp2_ep2_tp2; do echo "$c: $(grep -a -c 'step: ' $OUT/$c.log) steps, $(grep -a -ciE 'error|Traceback' $OUT/$c.log) error lines"; done
