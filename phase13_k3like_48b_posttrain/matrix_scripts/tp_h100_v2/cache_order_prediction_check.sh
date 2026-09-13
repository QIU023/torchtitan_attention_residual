#!/bin/bash
# Step-1 gradients, cache off vs on, bf16 standard path, at the two shapes not dumped before: pp2 x vp2 (1024 tokens,
# the H100 cell) and pp8 x vp2 (256 tokens, the precision triplet). The summation trees in PP_CACHE_REDUCTION_ORDER
# predict which parameters differ: pp2 x vp2 only tok_embeddings; pp8 x vp2 layers 0-11 and tok_embeddings.
set -u; export CFG=kimi_k3_debugmodel_c4 KDA_NOAUTOTUNE=1; . "$(dirname "$0")/common.sh"
J=/tmp/jitwarm/predcheck; mkdir -p $J/tmp; export TMPDIR=$J/tmp
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
g() { local nm=$1 gpus=$2 np=$3 common=$4 stag=$5; shift 5; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -al $OUT/seed_$stag/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton GRAD_DUMP=$OUT/grads_$nm \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $common --training.steps 1 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? files=$(ls $OUT/grads_$nm.rank*.pt 2>/dev/null | wc -l)"; rm -rf $d/checkpoint; }
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
seed c4_1024 $B4
g pc_vp2n 0,1 2 "$NAIVE" c4_1024 $B4 $D 1 $P2 &
g pc_vp2c 2,3 2 "$COMMON" c4_1024 $B4 $D 1 $P2 &
wait
export C4_ROW_TOKENS=64
B8="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 64"
P8="--parallelism.pipeline_parallel_degree 8 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
g pc_pp8vp2n 0,1,2,3,4,5,6,7 8 "$NAIVE" c4_256 $B8 $D 1 $P8
g pc_pp8vp2c 0,1,2,3,4,5,6,7 8 "$COMMON" c4_256 $B8 $D 1 $P8
for pair in "pc_vp2n pc_vp2c" "pc_pp8vp2n pc_pp8vp2c"; do set -- $pair; python3 "$(dirname "$0")/cache_order_compare.py" $OUT $1 $2 | head -30; done
rm -f $OUT/grads_pc_*
echo PREDCHECK-DONE
