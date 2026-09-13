#!/bin/bash
# Step-1 gradients, cache on vs off (pp4 x vp4) against dp1 no-sync, c4 1024 tokens, one seed, ONE shared
# inductor/triton cache so every cell uses the same autotuned kernels. Tree: pp_fp64_probe (c4 flavor,
# NOSYNC_GA, PP_STAGES_PER_RANK, GRAD_DUMP). Local 8 x 5060 Ti. Compare with cache_order_compare.py.
set -u; export CFG=kimi_k3_debugmodel_c4; . "$(dirname "$0")/common.sh"
J=/tmp/jitwarm/cache_order; mkdir -p $J/tmp; export TMPDIR=$J/tmp
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
seed c4_1024 $B4
g() {  # g <name> <gpus> <nproc> <flags...>: one step, gradients dumped, shared cache
  local nm=$1 gpus=$2 np=$3; shift 3; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r $OUT/seed_c4_1024/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton GRAD_DUMP=$OUT/grads_$nm \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $COMMON --training.steps 1 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? files=$(ls $OUT/grads_$nm.rank*.pt 2>/dev/null | wc -l)"; rm -rf $d/checkpoint
}
( export NOSYNC_GA=1; g dp1_ns 0 1 $B4 $D 1 )
( export PP_STAGES_PER_RANK=4; naive_common; g pp4vp4n 0,1,2,3 4 $B4 $D 1 $P4 )
( export PP_STAGES_PER_RANK=4; g pp4vp4c 0,1,2,3 4 $B4 $D 1 $P4 )
( export PP_STAGES_PER_RANK=4; g pp4vp4c_again 0,1,2,3 4 $B4 $D 1 $P4 )
( export PP_STAGES_PER_RANK=4; naive_common; g pp4vp4n_again 0,1,2,3 4 $B4 $D 1 $P4 )
echo GRADPROBE-DONE
