#!/bin/bash
# The DSV3 table's regime: float32 masters (training.dtype float32), bf16 compute, full 24-layer model, 100 steps.
set -u; . "$(dirname "$0")/common.sh"
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
seed fp32m $B
cell m_tp1_parent $TT_PARENT 0   1 fp32m 100 $B $D 1 $PD &
cell m_tp1        $TT        1   1 fp32m 100 $B $D 1 $PD &
cell m_tp1_st     $TT        2   1 fp32m 100 $B $D 1 $ST &
cell m_tp2_sp     $TT        3,4 2 fp32m 100 $B $D 1 $T 2 $PD &
cell m_tp2_sp_st  $TT        5,6 2 fp32m 100 $B $D 1 $T 2 $ST &
wait
cell m_tp2_nosp    $TT 0,1 2 fp32m 100 $B $D 1 $T 2 $PD $NOSP &
cell m_tp2_nosp_st $TT 2,3 2 fp32m 100 $B $D 1 $T 2 $ST $NOSP &
wait
table m_tp1_parent m_tp1 m_tp1_st m_tp2_sp m_tp2_sp_st m_tp2_nosp m_tp2_nosp_st
