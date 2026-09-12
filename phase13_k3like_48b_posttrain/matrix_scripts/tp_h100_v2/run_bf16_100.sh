#!/bin/bash
# TP/SP matrix for #4499 on 2 GPUs, 4500's format: bf16, 256 tokens per step, 100 steps,
# one seed checkpoint per batch shape, one inductor cache per cell. Reference: tp=1 on main.
# Cells that need four GPUs (dp2 x tp2, dp2 x ep2 x tp2, any tp=4) are not in this list.
set -u; . "$(dirname "$0")/common.sh"

B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
B2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
seed bf16 $B
seed bf16_dp2 $B2

# --- dp1 stream: the reference pair must be bitwise ---------------------------
cell tp1_parent   $TT_PARENT 0   1 bf16 100 $B $D 1 $ST
cell tp1          $TT        0   1 bf16 100 $B $D 1 $ST
cell tp1_again    $TT        1   1 bf16 100 $B $D 1 $ST          # noise floor: same cell, fresh cache
cell tp2_sp       $TT        0,1 2 bf16 100 $B $D 1 $T 2 $ST
cell tp2_nosp     $TT        0,1 2 bf16 100 $B $D 1 $T 2 $ST $NOSP

# the non-TP path is untouched: same pair on partial_dtensor
cell tp1_parent_pd $TT_PARENT 0  1 bf16 100 $B $D 1 $PD
cell tp1_pd        $TT        0  1 bf16 100 $B $D 1 $PD

# --- dp2 stream: its own reference (a second dp rank reads other samples) -----
cell dp2          $TT        0,1 2 bf16_dp2 100 $B2 $D 2 $ST
cell dp2_ep2      $TT        0,1 2 bf16_dp2 100 $B2 $D 2 $E 2 $ST

echo
echo '# dp1 stream (reference: tp=1 on main; tp1 must read bitwise with it)'
table tp1_parent tp1 tp1_again tp2_sp tp2_nosp
echo
echo '# the non-TP path, partial_dtensor (reference: tp=1 on main)'
table tp1_parent_pd tp1_pd
echo
echo '# dp2 stream (reference: dp2 on this branch)'
table dp2 dp2_ep2
