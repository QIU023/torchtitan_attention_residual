#!/bin/bash
# pp4 x vp4 cannot run at dp2 on four GPUs (dp2 x pp4 = 8), so the 2048-token step goes to dp1 as 8 x 256
# micro-batches: its own stream, own seed, own references (dp1, dp1 no-sync, dp1 reversed). Also c4_pp4vp4c
# again on a copy of its own inductor cache (determinism of the 1024 cache-on cell). Same OUT as run_pp_c4.sh.
set -u; export CFG=kimi_k3_debugmodel_c4; . "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
P4_1024="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
rm -rf $OUT/ind_c4_pp4vp4c_again $OUT/tri_c4_pp4vp4c_again
cp -r $OUT/ind_c4_pp4vp4c $OUT/ind_c4_pp4vp4c_again; cp -r $OUT/tri_c4_pp4vp4c $OUT/tri_c4_pp4vp4c_again
( export PP_STAGES_PER_RANK=4; cell c4_pp4vp4c_again $TT 0,1,2,3 4 c4_1024 100 $B4 $D 1 $P4_1024 )
seed c4_2048dp1 $B8
cell c4_w1_dp1 $TT 0 1 c4_2048dp1 100 $B8 $D 1 &
( export NOSYNC_GA=1; cell c4_w1_dp1_ns $TT 1 1 c4_2048dp1 100 $B8 $D 1 ) &
( export MB_REVERSE=1; cell c4_w1_dp1_rev $TT 2 1 c4_2048dp1 100 $B8 $D 1 ) &
wait
( export PP_STAGES_PER_RANK=4; cell c4_w1_pp4vp4c $TT 0,1,2,3 4 c4_2048dp1 100 $B8 $D 1 $P4 )
( export PP_STAGES_PER_RANK=4; naive_common; cell c4_w1_pp4vp4n $TT 0,1,2,3 4 c4_2048dp1 100 $B8 $D 1 $P4 )
export TABLE_STEPS="1 2 3 5 10 20 50 100"
echo; echo '# c4, 1024 tokens per step, reference dp1 no-sync: pp4 x vp4 cache on, twice'
table c4_dp1_ns c4_pp4vp4c c4_pp4vp4c_again c4_pp4vp4n
echo; echo '# c4, 2048 tokens per step at dp1 (8 x 256), reference dp1'
table c4_w1_dp1 c4_w1_dp1_rev c4_w1_dp1_ns c4_w1_pp4vp4c c4_w1_pp4vp4n
echo; echo '# the same, reference dp1 no-sync'
table c4_w1_dp1_ns c4_w1_pp4vp4c c4_w1_pp4vp4n
echo RUN-PP-C4-2048DP1-DONE
