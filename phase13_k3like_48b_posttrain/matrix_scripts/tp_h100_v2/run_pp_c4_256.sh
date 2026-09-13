#!/bin/bash
# PR 4312 (pp_review4 + pp4h_probe_c4_rowtokens.patch) in #4500's shape: c4_test as text-only 64-token rows
# (2000 rows, ~128k tokens), 256 tokens per step as four 64-token micro-batches (the pipeline needs several
# micro-batches per step; CP does not, so #4500 runs one 256-token micro-batch), dp2 at 512. 100 steps read
# 20 % / 40 % of the rows once. 4 GPUs. Set TT (pp_review4 + the rowtokens patch + pp_stages_per_rank.patch), OUT.
set -u; export CFG=kimi_k3_debugmodel_c4 C4_ROW_TOKENS=64; . "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 64"
B8="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 64"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 $IL"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
seed c4_256 $B4
seed c4_512 $B8
# 256 tokens per step (4 x 64), reference dp1
cell c4r64_dp1 $TT 0 1 c4_256 100 $B4 $D 1 &
( export MB_REVERSE=1; cell c4r64_dp1_rev $TT 1 1 c4_256 100 $B4 $D 1 ) &
cell c4r64_pp2 $TT 2,3 2 c4_256 100 $B4 $D 1 $P &
wait
cell c4r64_vp2c $TT 0,1 2 c4_256 100 $B4 $D 1 $P $IL &
( naive_common; cell c4r64_vp2n $TT 2,3 2 c4_256 100 $B4 $D 1 $P $IL ) &
wait
# pp4 x vp4 (16 stages), all four GPUs
( export PP_STAGES_PER_RANK=4; cell c4r64_pp4vp4c $TT 0,1,2,3 4 c4_256 100 $B4 $D 1 $P4 )
( export PP_STAGES_PER_RANK=4; naive_common; cell c4r64_pp4vp4n $TT 0,1,2,3 4 c4_256 100 $B4 $D 1 $P4 )
# 512 tokens per step (4 x 64 per rank), reference dp2
cell c4r64_d2_dp2 $TT 0,1 2 c4_512 100 $B8 $D 2 &
cell c4r64_d2_ep2 $TT 2,3 2 c4_512 100 $B8 $D 2 $E 2 &
wait
( export NOSYNC_GA=1; cell c4r64_dp1_ns $TT 0 1 c4_256 100 $B4 $D 1 ) &
wait
cell c4r64_d2_pp2 $TT 0,1,2,3 4 c4_512 100 $B8 $D 2 $P
cell c4r64_d2_vp2c $TT 0,1,2,3 4 c4_512 100 $B8 $D 2 $P $IL
( naive_common; cell c4r64_d2_vp2n $TT 0,1,2,3 4 c4_512 100 $B8 $D 2 $P $IL )
export TABLE_STEPS="1 10 20 50 100"  # which of these get reported is decided from the reference trajectory
echo; echo '# c4 (64-token rows), 256 tokens per step (4 x 64), reference dp1'
table c4r64_dp1 c4r64_dp1_rev c4r64_dp1_ns c4r64_pp2 c4r64_vp2c c4r64_vp2n c4r64_pp4vp4c c4r64_pp4vp4n
echo; echo '# c4 (64-token rows), 512 tokens per step (4 x 64 per rank), reference dp2'
table c4r64_d2_dp2 c4r64_d2_ep2 c4r64_d2_pp2 c4r64_d2_vp2c c4r64_d2_vp2n
echo RUN-PP-C4-256-DONE
