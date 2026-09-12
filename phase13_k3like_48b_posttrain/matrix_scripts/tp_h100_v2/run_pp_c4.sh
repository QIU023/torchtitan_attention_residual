#!/bin/bash
# PR 4312 (pp_review4 + pp4h_probe_c4.patch) on c4_test as text-only rows (each doc's first 256 tokens,
# 2000 rows, ~0.5M tokens): a 100-step run reads 21% (1024/step) or 43% (2048/step) of it once, so the
# reference cannot memorise it the way it memorises the 32-sample cc12m-test by step 20. 4 GPUs, 100 steps,
# one seed checkpoint per batch shape. Set TT (pp_review4 + probe patch), OUT.
set -u; export CFG=kimi_k3_debugmodel_c4; . "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
seed c4_1024 $B4
seed c4_2048 $B8
# 1024 tokens per step, reference dp1
cell c4_dp1 $TT 0 1 c4_1024 100 $B4 $D 1 &
( export MB_REVERSE=1; cell c4_dp1_rev $TT 1 1 c4_1024 100 $B4 $D 1 ) &
cell c4_pp2 $TT 2,3 2 c4_1024 100 $B4 $D 1 $P &
wait
cell c4_vp2c $TT 0,1 2 c4_1024 100 $B4 $D 1 $P $IL &
( naive_common; cell c4_vp2n $TT 2,3 2 c4_1024 100 $B4 $D 1 $P $IL ) &
wait
# 2048 tokens per step (4 x 256 per rank), reference dp2
cell c4_d2_dp2 $TT 0,1 2 c4_2048 100 $B8 $D 2 &
cell c4_d2_ep2 $TT 2,3 2 c4_2048 100 $B8 $D 2 $E 2 &
wait
( export NOSYNC_GA=1; cell c4_dp1_ns $TT 0 1 c4_1024 100 $B4 $D 1 ) &
wait
cell c4_d2_pp2 $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P
cell c4_d2_vp2c $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P $IL
( naive_common; cell c4_d2_vp2n $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P $IL )
export TABLE_STEPS="1 10 20 50 100"  # which of these get reported is decided from the reference trajectory
echo; echo '# c4, 1024 tokens per step (4 x 256), reference dp1'
table c4_dp1 c4_dp1_rev c4_dp1_ns c4_pp2 c4_vp2c c4_vp2n
echo; echo '# c4, 2048 tokens per step (4 x 256 per rank), reference dp2'
table c4_d2_dp2 c4_d2_ep2 c4_d2_pp2 c4_d2_vp2c c4_d2_vp2n
echo RUN-PP-C4-DONE
