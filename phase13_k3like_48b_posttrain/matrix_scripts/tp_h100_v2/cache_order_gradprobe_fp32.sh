#!/bin/bash
# cache_order_gradprobe.sh in float32: FP32_PROBE=1 keeps bf16 casts float32, loops the experts in float32 and
# runs KDA through Attention Gym's reference path; params, compute and reduce in float32. Local 5060 only.
set -u; export CFG=kimi_k3_debugmodel_c4 FP32_PROBE=1; . "$(dirname "$0")/common.sh"
J=/tmp/jitwarm/cache_order_fp32; mkdir -p $J/tmp; export TMPDIR=$J/tmp
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
F32="--training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
g() { local nm=$1 gpus=$2 np=$3 common=$4; shift 4; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r $OUT/seed_c4_1024/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton GRAD_DUMP=$OUT/grads_$nm \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $common --training.steps 1 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; echo "$nm rc=$rc files=$(ls $OUT/grads_$nm.rank*.pt 2>/dev/null | wc -l)"; rm -rf $d/checkpoint; return $rc; }
[ -n "${SKIP_DP1:-}" ] || ( export NOSYNC_GA=1; g fp32_dp1_ns 0 1 "$COMMON" $B4 $D 1 $F32 ) || { grep -a -E "Error" $OUT/fp32_dp1_ns.log | tail -3; exit 1; }
( export PP_STAGES_PER_RANK=4; g fp32_pp4vp4n 0,1,2,3 4 "$NAIVE" $B4 $D 1 $P4 $F32 )
( export PP_STAGES_PER_RANK=4; g fp32_pp4vp4c 0,1,2,3 4 "$COMMON" $B4 $D 1 $P4 $F32 )
echo GRADPROBE-FP32-DONE
