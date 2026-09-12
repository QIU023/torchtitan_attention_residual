#!/bin/bash
# The 1024-token PP cells with training.mixed_precision_reduce bfloat16 (upstream #4597): the reduce
# dtype equals the parameter dtype, so FSDP accumulates micro-batches in bf16 in both the pipeline
# (no-sync) and dp1 (sync each micro-batch) paths. cc12m-test, #4500's dataset. Waits for run_pp_cc12m.sh.
set -u
until grep -q RUN-PP-CC12M-DONE /workspace/pp_cc12m_run.log 2>/dev/null; do sleep 20; done
. "$(dirname "$0")/common.sh"
R="--training.mixed_precision_reduce bfloat16"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P4="$D 1 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel /--config kimi_k3_debugmodel_pp_naive }"; }
seed bf16_1024 $B4
cell br_dp1 $TT 0 1 bf16_1024 100 $B4 $R $D 1 &
( export NOSYNC_GA=1; cell br_dp1_ns $TT 1 1 bf16_1024 100 $B4 $R $D 1 ) &
cell br_pp2 $TT 2,3 2 bf16_1024 100 $B4 $R $P4 &
wait
cell br_vp2c $TT 0,1 2 bf16_1024 100 $B4 $R $P4 $IL &
( naive_common; cell br_vp2n $TT 2,3 2 bf16_1024 100 $B4 $R $P4 $IL ) &
wait
( export MB_REVERSE=1; cell br_dp1_rev $TT 0 1 bf16_1024 100 $B4 $R $D 1 )
echo
echo '# mixed_precision_reduce bfloat16, cc12m-test, 1024 tokens per step (4 x 256), reference dp1'
table br_dp1 br_dp1_ns br_dp1_rev br_pp2 br_vp2c br_vp2n
echo RUN-PP-BF16R-DONE
