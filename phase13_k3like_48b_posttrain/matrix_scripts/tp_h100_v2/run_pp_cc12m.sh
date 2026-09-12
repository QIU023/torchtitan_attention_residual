#!/bin/bash
# The 1024-token PP cells again on the streamed cc12m (no sample repeats in 100 steps), so step 100
# compares configurations instead of memorised curves. Not #4500's dataset. Waits for run_pp.sh.
set -u
until grep -q RUN-PP-DONE /workspace/pp_run.log 2>/dev/null; do sleep 20; done
CFG=kimi_k3_debugmodel_cc12m
. "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P4="$D 1 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_cc12m /--config kimi_k3_debugmodel_cc12m_pp_naive }"; }
seed cc_1024 $B4
cell cc_dp1 $TT 0 1 cc_1024 100 $B4 $D 1 &
( export NOSYNC_GA=1; cell cc_dp1_ns $TT 1 1 cc_1024 100 $B4 $D 1 ) &
cell cc_pp2 $TT 2,3 2 cc_1024 100 $B4 $P4 &
wait
cell cc_vp2c $TT 0,1 2 cc_1024 100 $B4 $P4 $IL &
( naive_common; cell cc_vp2n $TT 2,3 2 cc_1024 100 $B4 $P4 $IL ) &
wait
( export MB_REVERSE=1; cell cc_dp1_rev $TT 0 1 cc_1024 100 $B4 $D 1 )
echo
echo '# streamed cc12m, 1024 tokens per step (4 x 256), reference dp1'
table cc_dp1 cc_dp1_ns cc_dp1_rev cc_pp2 cc_vp2c cc_vp2n
echo
echo '# the same pipeline cells against dp1 accumulating like the pipeline'
table cc_dp1_ns cc_pp2 cc_vp2c cc_vp2n
echo RUN-PP-CC12M-DONE
