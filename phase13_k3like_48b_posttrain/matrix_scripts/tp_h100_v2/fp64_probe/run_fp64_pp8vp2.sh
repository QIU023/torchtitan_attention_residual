#!/bin/bash
# float64 trajectories, cache off then cache on, pp8 x vp2 (16 stages, two per rank: pp4 x vp4 in fp64 runs out of
# memory on the rank holding lm_head at the first optimizer step). c4 64-token rows, 256 tokens per step as 4 x 64,
# one seed, full-precision loss / norm every step. All eight GPUs per cell, so the two cells run one after the other.
set -u; export CFG=kimi_k3_debugmodel_c4 C4_ROW_TOKENS=64 FP64_PROBE=1 REPR_LOG=1
. "$(dirname "$0")/../common.sh"
STEPS=${STEPS:-20}
J=/tmp/jitwarm/fp64; mkdir -p $J/tmp; export TMPDIR=$J/tmp
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 64"
F64="--training.dtype float64 --training.mixed_precision_param float64 --training.mixed_precision_reduce float64"
P8="--parallelism.pipeline_parallel_degree 8 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
seed c4_256 $B
g() { local nm=$1 common=$2; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -al $OUT/seed_c4_256/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton \
      torchrun --nproc_per_node=8 --master_port=$((30000+RANDOM%20000)) $common --training.steps $STEPS $B $D 1 $P8 $F64 --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? repr_steps=$(grep -c '^REPR [0-9]* rank7' $OUT/$nm.log) $(date +%T)"; rm -rf $d/checkpoint; }
g fp64_pp8vp2n "$NAIVE"
g fp64_pp8vp2c "$COMMON"
python3 "$(dirname "$0")/repr_compare.py" $OUT/fp64_pp8vp2n.log $OUT/fp64_pp8vp2c.log
echo FP64-PP8-DONE
