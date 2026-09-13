#!/bin/bash
# pp8 x vp2, cache off vs cache on, in bf16 (the standard path, no probe switch), fp32 (FP32_PROBE) and fp64
# (FP64_PROBE): c4 64-token rows, 256 tokens per step as 4 x 64, one fp32 seed, 100 steps, full-precision loss and
# norm every step (REPR_LOG, logging only). Per precision one shared compile cache, warmed by a 1-step run of both
# cells first. KDA_NOAUTOTUNE=1 keeps the fused KDA's configs fixed in bf16 (the other two run the reference KDA).
# Local 8 x 5060 Ti, every cell on all eight GPUs, one after the other. Tree: pp_fp64_probe.
set -u; export CFG=kimi_k3_debugmodel_c4 C4_ROW_TOKENS=64 REPR_LOG=1 KDA_NOAUTOTUNE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. "$(dirname "$0")/../common.sh"
STEPS=${STEPS:-100}; PRECS=${PRECS:-"bf16 fp32 fp64"}
TAG=${TAG:-}; EXTRA=${EXTRA:-}  # e.g. TAG=_clr EXTRA="--lr_scheduler.min_lr_factor 1.0" for a constant LR after the 2 warmup steps
B="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 64"
P8="--parallelism.pipeline_parallel_degree 8 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
seed c4_256 $B
g() { local nm=$1 common=$2 steps=$3 flags=$4; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -al $OUT/seed_c4_256/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton \
      torchrun --nproc_per_node=8 --master_port=$((30000+RANDOM%20000)) $common --training.steps $steps $B $D 1 $P8 $flags $EXTRA --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? $(date +%T)"; rm -rf $d/checkpoint; }
for prec in $PRECS; do
  unset FP32_PROBE FP64_PROBE
  case $prec in
    bf16) flags="" ;;
    fp32) export FP32_PROBE=1; flags="--training.dtype float32 --training.mixed_precision_param float32 --training.mixed_precision_reduce float32" ;;
    fp64) export FP64_PROBE=1; flags="--training.dtype float64 --training.mixed_precision_param float64 --training.mixed_precision_reduce float64" ;;
  esac
  J=/tmp/jitwarm/triplet_$prec; mkdir -p $J/tmp; export TMPDIR=$J/tmp
  g tri${TAG}_${prec}_warm_n "$NAIVE" 1 "$flags"; g tri${TAG}_${prec}_warm_c "$COMMON" 1 "$flags"
  g tri${TAG}_${prec}_n "$NAIVE" $STEPS "$flags"; g tri${TAG}_${prec}_c "$COMMON" $STEPS "$flags"
  python3 "$(dirname "$0")/repr_compare.py" $OUT/tri${TAG}_${prec}_n.log $OUT/tri${TAG}_${prec}_c.log > $OUT/tri${TAG}_${prec}_compare.txt
  echo "== $prec: $(wc -l < $OUT/tri${TAG}_${prec}_compare.txt) steps compared"; sed -n '1p;10p;20p;50p;100p' $OUT/tri${TAG}_${prec}_compare.txt | cut -c1-200
done
echo TRIPLET-DONE
