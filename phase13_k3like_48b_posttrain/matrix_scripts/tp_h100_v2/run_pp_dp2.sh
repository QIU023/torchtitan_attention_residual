#!/bin/bash
# PR 4312: the dp2 stream on cc12m-test, 2048 tokens per step (4 x 256 per rank; pp2 x vp2 needs four
# micro-batches), 100 steps, reference dp2. Waits for run_pp_bf16r.sh. Set TT (pp_review4 + probe patch), OUT.
set -u
until grep -q RUN-PP-BF16R-DONE /workspace/pp_bf16r_run.log 2>/dev/null; do sleep 20; done
. "$(dirname "$0")/common.sh"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
D2="$D 2"; P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel /--config kimi_k3_debugmodel_pp_naive }"; }
seed bf16_2048 $B8
cell d2_dp2 $TT 0,1 2 bf16_2048 100 $B8 $D2 &
cell d2_ep2 $TT 2,3 2 bf16_2048 100 $B8 $D2 $E 2 &
wait
( export NOSYNC_GA=1; cell d2_dp2_ns $TT 0,1 2 bf16_2048 100 $B8 $D2 ) &
( export MB_REVERSE=1; cell d2_dp2_rev $TT 2,3 2 bf16_2048 100 $B8 $D2 ) &
wait
cell d2_pp2 $TT 0,1,2,3 4 bf16_2048 100 $B8 $D2 $P
cell d2_vp2c $TT 0,1,2,3 4 bf16_2048 100 $B8 $D2 $P $IL
( naive_common; cell d2_vp2n $TT 0,1,2,3 4 bf16_2048 100 $B8 $D2 $P $IL )
echo
echo '# dp2 stream, cc12m-test, 2048 tokens per step (4 x 256 per rank), reference dp2'
table d2_dp2 d2_dp2_ns d2_dp2_rev d2_ep2 d2_pp2 d2_vp2c d2_vp2n
echo
echo '# the pipeline cells against dp2 accumulating like the pipeline'
table d2_dp2_ns d2_pp2 d2_vp2c d2_vp2n
echo RUN-PP-DP2-DONE
