#!/bin/bash
# float64 end to end (FP64_PROBE=1 on pp_fp64_probe: casts, KDA + short conv on Attention Gym's oracles, eager flex,
# per-expert MoE loop, fp64 loss accumulation), pp4 x vp4 cache off vs cache on, c4 64-token rows, 256 tokens per
# step as 4 x 64, one seed, full-precision loss / norm every step (REPR_LOG). Local 8 x 5060 Ti. STEPS (default 2)
# and DUMP=1 (step-1 gradients) from the environment. Naive on GPUs 0-3, cached on 4-7, one shared cache.
set -u; export CFG=kimi_k3_debugmodel_c4 C4_ROW_TOKENS=64 FP64_PROBE=1 REPR_LOG=1 PP_STAGES_PER_RANK=4
. "$(dirname "$0")/../common.sh"
STEPS=${STEPS:-2}
J=/tmp/jitwarm/fp64; mkdir -p $J/tmp; export TMPDIR=$J/tmp
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 64"
F64="--training.dtype float64 --training.mixed_precision_param float64 --training.mixed_precision_reduce float64"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
seed c4_256 $B
g() { local nm=$1 gpus=$2 common=$3; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -al $OUT/seed_c4_256/checkpoint $d/
  local dump=""; [ "${DUMP:-0}" = 1 ] && dump="GRAD_DUMP=$OUT/grads_$nm"
  ( cd $TT && env $dump CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton \
      torchrun --nproc_per_node=4 --master_port=$((30000+RANDOM%20000)) $common --training.steps $STEPS $B $D 1 $P4 $F64 --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? repr=$(grep -c '^REPR' $OUT/$nm.log)"; rm -rf $d/checkpoint; }
g fp64_pp4vp4n 0,1,2,3 "$NAIVE" &
g fp64_pp4vp4c 4,5,6,7 "$COMMON" &
wait
echo FP64-DONE
