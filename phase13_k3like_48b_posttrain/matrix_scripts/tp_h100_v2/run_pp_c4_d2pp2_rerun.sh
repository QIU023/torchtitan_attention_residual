#!/bin/bash
# dp2 x pp2 (1F1B) differs from the dp2 no-sync reference from step 1 with the fp32 grad norm, while dp2 x pp2 x vp2
# naive is identical. (1) the same cell again on a copy of its own cache; (2) step-1 gradients of the reference,
# dp2 x pp2 and dp2 x pp2 x vp2 naive on one shared cache. GN_FP32=1 throughout. OUT = run_pp_c4.sh's.
set -u; export CFG=kimi_k3_debugmodel_c4 GN_FP32=1; . "$(dirname "$0")/common.sh"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
rm -rf $OUT/ind_gn32_c4_d2_pp2_again $OUT/tri_gn32_c4_d2_pp2_again
cp -r $OUT/ind_gn32_c4_d2_pp2 $OUT/ind_gn32_c4_d2_pp2_again; cp -r $OUT/tri_gn32_c4_d2_pp2 $OUT/tri_gn32_c4_d2_pp2_again
cell gn32_c4_d2_pp2_again $TT 0,1,2,3 4 c4_2048 100 $B8 $D 2 $P
J=$OUT/jitwarm_gd; mkdir -p $J
g() { local nm=$1 gpus=$2 np=$3 common=$4; shift 4; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r $OUT/seed_c4_2048/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton GRAD_DUMP=$OUT/grads_$nm \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $common --training.steps 1 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? files=$(ls $OUT/grads_$nm.rank*.pt 2>/dev/null | wc -l)"; rm -rf $d/checkpoint; }
( export NOSYNC_GA=1; g gd_d2_dp2_ns 0,1 2 "$COMMON" $B8 $D 2 )
g gd_d2_pp2 0,1,2,3 4 "$COMMON" $B8 $D 2 $P
g gd_d2_vp2n 0,1,2,3 4 "$NAIVE" $B8 $D 2 $P $IL
export TABLE_STEPS="1 2 3 10 20 100"
echo; table gn32_c4_d2_dp2_ns gn32_c4_d2_pp2 gn32_c4_d2_pp2_again gn32_c4_d2_vp2n
python3 "$(dirname "$0")/cache_order_compare.py" $OUT gd_d2_dp2_ns gd_d2_pp2 gd_d2_vp2n
echo RUN-D2PP2-RERUN-DONE
