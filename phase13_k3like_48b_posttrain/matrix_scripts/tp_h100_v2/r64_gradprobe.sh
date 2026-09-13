#!/bin/bash
# Step-1 gradients at #4500's shape (c4 64-token rows, 256 tokens per step as 4 x 64), local 5060 only: where does the
# interleaved cells' 1-ulp step-1 grad norm gap (H100 256 kit) come from? One seed, one shared cache.
set -u; export CFG=kimi_k3_debugmodel_c4 C4_ROW_TOKENS=64; . "$(dirname "$0")/common.sh"
J=/tmp/jitwarm/r64; mkdir -p $J/tmp; export TMPDIR=$J/tmp
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 64"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
seed c4_256 $B
g() { local nm=$1 gpus=$2 np=$3 common=$4; shift 4; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r $OUT/seed_c4_256/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton GRAD_DUMP=$OUT/grads_$nm \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $common --training.steps 1 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? files=$(ls $OUT/grads_$nm.rank*.pt 2>/dev/null | wc -l) $(grep -a -E 'step: *1 ' $OUT/$nm.log | sed -E 's/\x1b\[[0-9;]*m//g' | grep -oE 'loss: *[0-9.]+ +grad_norm: *[0-9.]+' | head -1)"; rm -rf $d/checkpoint; }
( export NOSYNC_GA=1; g r64_dp1_ns 0 1 "$COMMON" $B $D 1 )
g r64_pp2 0,1 2 "$COMMON" $B $D 1 $P
g r64_vp2n 0,1 2 "$NAIVE" $B $D 1 $P $IL
g r64_vp2c 0,1 2 "$COMMON" $B $D 1 $P $IL
echo R64-GRADPROBE-DONE
