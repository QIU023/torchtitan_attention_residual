#!/bin/bash
# run_pp_c4.sh's 2048-token dp2 stream with the total grad norm in float32 (GN_FP32=1, gn_fp32_hack.py). Same seed.
set -u; export CFG=kimi_k3_debugmodel_c4 GN_FP32=1; . "$(dirname "$0")/common.sh"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
( export NOSYNC_GA=1; cell gn32_c4_d2_dp2_ns $TT 0,1 2 c4_2048 100 $B8 $D 2 ) &
( export NOSYNC_GA=1 MB_REVERSE=1; cell gn32_c4_d2_dp2_ns_rev $TT 2,3 2 c4_2048 100 $B8 $D 2 ) &
wait
cell gn32_c4_d2_pp2 $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P
cell gn32_c4_d2_vp2c $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P $IL
( naive_common; cell gn32_c4_d2_vp2n $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P $IL )
export TABLE_STEPS="1 2 3 5 10 20 100"
echo; echo '# c4 2048 dp2, grad norm in fp32, reference dp2 no-sync (fp32 norm)'
table gn32_c4_d2_dp2_ns gn32_c4_d2_dp2_ns_rev gn32_c4_d2_pp2 gn32_c4_d2_vp2c gn32_c4_d2_vp2n
echo RUN-PP-C4-D2-GN32-DONE
