#!/bin/bash
# The 1024-token PP cells with Attention Gym's fused KDA autotune off (KDA_NOAUTOTUNE=1: chunk_kda's
# autotune=False; the kernels' own @triton.autotune still runs). cc12m-test. Waits for run_pp_dp2.sh.
set -u
until grep -q RUN-PP-DP2-DONE /workspace/pp_dp2_run.log 2>/dev/null; do sleep 20; done
export KDA_NOAUTOTUNE=1
. "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P4="$D 1 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel /--config kimi_k3_debugmodel_pp_naive }"; }
seed bf16_1024 $B4
cell ka_dp1 $TT 0 1 bf16_1024 100 $B4 $D 1 &
( export NOSYNC_GA=1; cell ka_dp1_ns $TT 1 1 bf16_1024 100 $B4 $D 1 ) &
cell ka_pp2 $TT 2,3 2 bf16_1024 100 $B4 $P4 &
wait
cell ka_vp2c $TT 0,1 2 bf16_1024 100 $B4 $P4 $IL &
( naive_common; cell ka_vp2n $TT 2,3 2 bf16_1024 100 $B4 $P4 $IL ) &
wait
echo
echo '# KDA autotune off, cc12m-test, 1024 tokens per step (4 x 256), reference dp1'
table ka_dp1 ka_dp1_ns ka_pp2 ka_vp2c ka_vp2n
echo
echo '# against dp1 accumulating like the pipeline'
table ka_dp1_ns ka_pp2 ka_vp2c ka_vp2n
echo RUN-PP-KDANOAT-DONE
