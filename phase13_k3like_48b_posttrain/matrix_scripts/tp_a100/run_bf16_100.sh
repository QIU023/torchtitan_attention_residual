#!/bin/bash
# 4500's table: bf16 flavor, 256 tokens per step, 100 steps; parent tp=1 is the reference.
set -u; . "$(dirname "$0")/common.sh"
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
seed bf16 $B
cell tp1_parent  $TT_PARENT 0   1 bf16 100 $B $D 1 $PD &
cell tp1         $TT        1   1 bf16 100 $B $D 1 $PD &
cell tp1_st      $TT        2   1 bf16 100 $B $D 1 $ST &
cell tp2_sp      $TT        3,4 2 bf16 100 $B $D 1 $T 2 $PD &
cell tp2_sp_st   $TT        5,6 2 bf16 100 $B $D 1 $T 2 $ST &
wait
cell tp1_parent_st $TT_PARENT 0 1 bf16 100 $B $D 1 $ST &
cell tp2_nosp    $TT        1,2 2 bf16 100 $B $D 1 $T 2 $PD $NOSP &
cell tp2_nosp_st $TT        3,4 2 bf16 100 $B $D 1 $T 2 $ST $NOSP &
wait
cell tp4_sp      $TT        0,1,2,3 4 bf16 100 $B $D 1 $T 4 $PD &
cell tp4_sp_st   $TT        4,5,6,7 4 bf16 100 $B $D 1 $T 4 $ST &
wait
cell tp4_nosp    $TT        0,1,2,3 4 bf16 100 $B $D 1 $T 4 $PD $NOSP &
cell tp4_nosp_st $TT        4,5,6,7 4 bf16 100 $B $D 1 $T 4 $ST $NOSP &
wait
table tp1_parent tp1_parent_st tp1 tp1_st tp2_sp tp2_sp_st tp2_nosp tp2_nosp_st tp4_sp tp4_sp_st tp4_nosp tp4_nosp_st
