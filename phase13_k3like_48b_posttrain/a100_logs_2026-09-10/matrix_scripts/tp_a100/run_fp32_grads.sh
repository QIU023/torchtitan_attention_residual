#!/bin/bash
# True float32 step-1 gradient dumps: dp1, the tp2 cells, and dp1 with the KDA projections perturbed by the colwise matmul's roundoff (the floor).
set -u; . "$(dirname "$0")/common.sh"
HK="$(cd "$(dirname "$0")" && pwd)/hacks"
X=$OUT/tree_fp32g; [ -d $X ] || { git -C $TT worktree add --detach $X HEAD >/dev/null; python $HK/experts_fp32_hack.py $X; python $HK/grad_tensor_dump_fp32_exit_hack.py $X; python $HK/perturb_control_hack.py $X; }
export EXPERTS_FP32=1 GRAD_TENSOR_DUMP_EXIT=1
B="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32"
seed fp32x $B
G=$OUT/grads; mkdir -p $G
GRAD_TENSOR_DUMP=$G/dp1        cell g_dp1        $X 0   1 fp32x 1 $B $D 1 $PD &
GRAD_TENSOR_DUMP=$G/dp1pert KDA_PERTURB=3e-7 cell g_dp1pert $X 1 1 fp32x 1 $B $D 1 $PD &
GRAD_TENSOR_DUMP=$G/tp2sp      cell g_tp2sp      $X 2,3 2 fp32x 1 $B $D 1 $T 2 $PD &
GRAD_TENSOR_DUMP=$G/tp2nosp    cell g_tp2nosp    $X 4,5 2 fp32x 1 $B $D 1 $T 2 $PD $NOSP &
GRAD_TENSOR_DUMP=$G/tp2sp_st   cell g_tp2sp_st   $X 6,7 2 fp32x 1 $B $D 1 $T 2 $ST &
wait
GRAD_TENSOR_DUMP=$G/tp2nosp_st cell g_tp2nosp_st $X 0,1 2 fp32x 1 $B $D 1 $T 2 $ST $NOSP
python3 "$(dirname "$0")/cmp_grad_dist.py" $G dp1 dp1pert tp2sp tp2nosp tp2sp_st tp2nosp_st
