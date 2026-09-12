#!/bin/bash
# Second half on the 4-GPU box, after run_v2.sh: the dp2 stream (its own table: 512 tokens per step,
# 256 per rank, since a second data-parallel rank reads other samples) and Kimi K2.5 at tp=1 against
# main (the PR touches kimi_k2_7; K2.5 refuses tp > 1 on main, DistMuon #3353). Set TT, TT_PARENT, OUT.
set -u; . "$(dirname "$0")/common.sh"
B2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
seed bf16_dp2 $B2
cell dp2          $TT 0,1     2 bf16_dp2 100 $B2 $D 2 &
cell dp2_ep2      $TT 2,3     2 bf16_dp2 100 $B2 $D 2 $E 2 &
wait
cell dp2_tp2      $TT 0,1,2,3 4 bf16_dp2 100 $B2 $D 2 $T 2
cell dp2_ep2_tp2  $TT 0,1,2,3 4 bf16_dp2 100 $B2 $D 2 $E 2 $T 2
echo
echo '# dp2 stream, 512 tokens per step (256 per rank; reference: dp2 on this PR)'
table dp2 dp2_ep2 dp2_tp2 dp2_ep2_tp2
echo RUN-V2-DP2-DONE
