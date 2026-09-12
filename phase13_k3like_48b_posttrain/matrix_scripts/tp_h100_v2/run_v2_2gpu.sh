#!/bin/bash
# The v2 table on a 2-GPU box: the same cells as run_v2.sh, ordered for two GPUs; the 4-GPU smoke
# runs only when four GPUs are visible. Set TT, TT_PARENT, OUT; PYTHONPATH carries the per-rank
# Triton cache shim and Attention Gym main.
set -u; . "$(dirname "$0")/common.sh"
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
B2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
seed bf16 $B
cell tp1_parent $TT_PARENT 0   1 bf16 100 $B $D 1 &
cell tp1        $TT        1   1 bf16 100 $B $D 1 &
wait
cell tp2_sp     $TT        0,1 2 bf16 100 $B $D 1 $T 2
cell tp2_nosp   $TT        0,1 2 bf16 100 $B $D 1 $T 2 $NOSP
cell tp1_again  $TT        0   1 bf16 100 $B $D 1
TAIL=activation_checkpoint:none cell tc_tp2 $TT 0,1 2 bf16 3 $B $D 1 $T 2 $TC
if [ "$(nvidia-smi -L | wc -l)" -ge 4 ]; then
  seed bf16_dp2 $B2
  TAIL=activation_checkpoint:none cell tc_dp2_ep2_tp2 $TT 0,1,2,3 4 bf16_dp2 3 $B2 $D 2 $E 2 $T 2 $TC
fi
echo
echo '# dp1, 256 tokens per step (reference: tp=1 on main; tp1 and tp1_again must be bitwise with it)'
table tp1_parent tp1 tp1_again tp2_sp tp2_nosp
echo
for c in tc_tp2 tc_dp2_ep2_tp2; do [ -f $OUT/$c.log ] && echo "$c: $(grep -a -c 'step: ' $OUT/$c.log) steps, $(grep -a -ciE 'Traceback' $OUT/$c.log) tracebacks"; done
echo RUN-V2-DONE
