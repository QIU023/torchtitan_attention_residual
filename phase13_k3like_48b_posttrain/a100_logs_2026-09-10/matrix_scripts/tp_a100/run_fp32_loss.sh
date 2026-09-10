#!/bin/bash
# True float32: masters and compute (mixed_precision_param/reduce float32) plus the fp32 experts loop; 2 steps; loss and grad norm per cell.
set -u; . "$(dirname "$0")/common.sh"
HK="$(cd "$(dirname "$0")" && pwd)/hacks"
X=$OUT/tree_fp32; [ -d $X ] || { git -C $TT worktree add --detach $X HEAD >/dev/null; python $HK/experts_fp32_hack.py $X; }
export EXPERTS_FP32=1
B="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32"
seed fp32x $B
cell x_dp1         $X 0   1 fp32x 2 $B $D 1 $PD &
cell x_tp2_sp      $X 1,2 2 fp32x 2 $B $D 1 $T 2 $PD &
cell x_tp2_sp_st   $X 3,4 2 fp32x 2 $B $D 1 $T 2 $ST &
cell x_tp2_nosp    $X 5,6 2 fp32x 2 $B $D 1 $T 2 $PD $NOSP &
wait
cell x_tp2_nosp_st $X 0,1 2 fp32x 2 $B $D 1 $T 2 $ST $NOSP
for nm in x_dp1 x_tp2_sp x_tp2_sp_st x_tp2_nosp x_tp2_nosp_st; do echo "$nm: $(grep -a -o 'step: *[0-9]*  .*loss: *[0-9.-]*  .*grad_norm: *[0-9.]*' $OUT/$nm.log | sed 's/\x1b\[[0-9;]*m//g' | grep -v 'loss: -1' | awk '{printf "s%s L=%s G=%s | ", $2, $4, $6}')"; done
