#!/bin/bash
# More order-only samples for the 2048-token dp2 stream, to size the spread the cached cell's step-100 -2.08% is read
# against: dp2 no-sync reversed, and fresh-inductor-cache reruns of dp2 no-sync, cached and naive dp2 x pp2 x vp2
# (a new cell name is a new cache; MB_REVERSE is a no-op under PP, one group per step). Same OUT as run_pp_c4.sh.
set -u; export CFG=kimi_k3_debugmodel_c4; . "$(dirname "$0")/common.sh"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
( export NOSYNC_GA=1 MB_REVERSE=1; cell c4_d2_dp2_ns_rev $TT 0,1 2 c4_2048 100 $B8 $D 2 ) &
( export NOSYNC_GA=1; cell c4_d2_dp2_ns_cold $TT 2,3 2 c4_2048 100 $B8 $D 2 ) &
wait
cell c4_d2_vp2c_cold $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P $IL
( naive_common; cell c4_d2_vp2n_cold $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P $IL )
export TABLE_STEPS="1 10 20 50 100"
echo; echo '# c4 2048 dp2 stream, reference dp2 no-sync: every order-only sample'
table c4_d2_dp2_ns c4_d2_dp2_ns_rev c4_d2_dp2_ns_cold c4_d2_dp2 c4_d2_dp2_rev c4_d2_ep2 c4_d2_pp2 c4_d2_pp2_again c4_d2_vp2c c4_d2_vp2c_cold c4_d2_vp2n c4_d2_vp2n_cold
echo RUN-PP-C4-D2-FLOOR-DONE
