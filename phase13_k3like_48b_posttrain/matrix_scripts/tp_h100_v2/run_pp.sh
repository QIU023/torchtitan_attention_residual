#!/bin/bash
# PR 4312 (pp_review4) in #4500's protocol on 4 GPUs, pp8 x vp4 left out. #4500's 256 tokens per step
# cannot hold a pipeline: the collator needs 256 per micro-batch and the pipeline one micro-batch per
# stage, so pp2 runs at 1024 (4 x 256) and 512 (2 x 256), pp2 x vp2 at 1024. Set TT (pp_review4 + probe patch), OUT.
set -u; . "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
B2="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
P4="$D 1 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
P2="$D 1 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 2"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel /--config kimi_k3_debugmodel_pp_naive }"; }
seed bf16_1024 $B4
seed bf16_512 $B2
# wave 1
cell pp_dp1 $TT 0 1 bf16_1024 100 $B4 $D 1 &
( export NOSYNC_GA=1; cell pp_dp1_ns $TT 1 1 bf16_1024 100 $B4 $D 1 ) &
cell pp_pp2 $TT 2,3 2 bf16_1024 100 $B4 $P4 &
wait
# wave 2
cell pp_vp2c $TT 0,1 2 bf16_1024 100 $B4 $P4 $IL &
( naive_common; cell pp_vp2n $TT 2,3 2 bf16_1024 100 $B4 $P4 $IL ) &
wait
# wave 3
( export MB_REVERSE=1; cell pp_dp1_rev $TT 0 1 bf16_1024 100 $B4 $D 1 ) &
cell pp_dp1_512 $TT 1 1 bf16_512 100 $B2 $D 1 &
cell pp_pp2_512 $TT 2,3 2 bf16_512 100 $B2 $P2 &
wait
echo
echo '# 1024 tokens per step (4 x 256), reference dp1 (micro-batches synced each step, bf16 accumulation)'
table pp_dp1 pp_dp1_ns pp_dp1_rev pp_pp2 pp_vp2c pp_vp2n
echo
echo '# the same pipeline cells against dp1 accumulating like the pipeline (FSDP no-sync, float32 accumulation)'
table pp_dp1_ns pp_pp2 pp_vp2c pp_vp2n
echo
echo '# 512 tokens per step (2 x 256): a two-term accumulation, exact either way'
table pp_dp1_512 pp_pp2_512
echo RUN-PP-DONE
