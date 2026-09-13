#!/bin/bash
# The forward with the rank cache on and off, on many inputs and many parameter draws, bf16 standard path, pp2 x vp2,
# 1024 tokens (4 x 256). (1) NO_OPT_STEP=1: parameters never change, 50 steps = 50 different batches on one parameter
# set; (2) seeds 1-4, no seed checkpoint (fresh init each), one step. Full-precision loss every step (REPR_LOG).
# One shared compile cache, warmed first; cells one after the other. Tree: pp_fp64_probe.
set -u; export CFG=kimi_k3_debugmodel_c4 REPR_LOG=1 KDA_NOAUTOTUNE=1; . "$(dirname "$0")/../common.sh"
J=/tmp/jitwarm/fwdcheck; mkdir -p $J/tmp; export TMPDIR=$J/tmp
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
seed c4_1024 $B4
g() { local nm=$1 common=$2 steps=$3 useseed=$4; shift 4; local d=$OUT/$nm; rm -rf $d; mkdir -p $d
  [ "$useseed" = 1 ] && cp -al $OUT/seed_c4_1024/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton \
      torchrun --nproc_per_node=2 --master_port=$((30000+RANDOM%20000)) $common --training.steps $steps $B4 $D 1 $P2 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$?"; rm -rf $d/checkpoint; }
g fw_warm_n "$NAIVE" 1 1; g fw_warm_c "$COMMON" 1 1
export NO_OPT_STEP=1
g fw_frozen_n "$NAIVE" 50 1; g fw_frozen_c "$COMMON" 50 1
unset NO_OPT_STEP
for k in 1 2 3 4; do g fw_seed${k}_n "$NAIVE" 1 0 --debug.seed $k; g fw_seed${k}_c "$COMMON" 1 0 --debug.seed $k; done
C="$(dirname "$0")/repr_compare.py"
echo "== frozen parameters, 50 batches:"; python3 $C $OUT/fw_frozen_n.log $OUT/fw_frozen_c.log > $OUT/fw_frozen_compare.txt
echo "  steps compared $(wc -l < $OUT/fw_frozen_compare.txt), steps with a loss difference $(grep -vc 'loss \S* vs \S* rel +0.00e+00' $OUT/fw_frozen_compare.txt), distinct losses $(awk '{print $4}' $OUT/fw_frozen_compare.txt | sort -u | wc -l)"
sed -n '1p;25p;50p' $OUT/fw_frozen_compare.txt | cut -c1-110
echo "== fresh init per seed, step 1:"
for k in 1 2 3 4; do python3 $C $OUT/fw_seed${k}_n.log $OUT/fw_seed${k}_c.log | sed "s/^/  seed $k: /" | cut -c1-120; done
echo FWDCHECK-DONE
