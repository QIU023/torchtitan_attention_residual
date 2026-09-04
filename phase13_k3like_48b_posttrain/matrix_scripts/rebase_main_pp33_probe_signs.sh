#!/bin/bash
# Step-1 gradient sign census on the rebased pp_review3 (0e7cc5ea1): dp1 on two fresh inductor
# caches (the same-cell control), pp2 x vp4 and pp8 x vp4 (cross-cell), 10 steps each so the
# trajectories are read from the same runs. Dumps are bf16 and deleted after the census.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
MS=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
T=/tmp/wt_ppprobe33; SEED=$(ls -d /workspace/.mx3_seeds_main33/kimi_k3_debugmodel_*/seed_ckpt | head -1)
OUT=/workspace/ppprobe33_$(date +%m%d_%H%M); mkdir -p $OUT; DUMP=$OUT/dumps; mkdir -p $DUMP
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
run(){ local nm=$1 np=$2 cache=$3; shift 3
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  ( source /venv/main/bin/activate && cd $T && PYTHONPATH=$T TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$cache TRITON_CACHE_DIR=$OUT/tri_$cache \
    GRAD_TENSOR_DUMP=$DUMP/$nm timeout 2400 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps 10 $B --checkpoint.enable --checkpoint.interval 100000 $D "$@" \
    --dump-folder $d > $OUT/$nm.log 2>&1 ); rm -rf $d/checkpoint
  printf "%-10s " $nm; sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -oE "step: +[0-9]+ +loss: +[0-9.]+ +grad_norm: +[0-9.]+" | awk '{printf "%s:%s/%s ", $2, $4, $6}'; echo
}
run dp1_A 1 A
run pp2_vp4 2 C $P 2 $L 4 $IL
run pp8_vp4 8 E $P 8 $L 1 $IL
source /venv/main/bin/activate
echo "
python $MS/pp_step10_census.py "dp1 vs pp2 x vp4" $DUMP/dp1_A $DUMP/pp2_vp4
python $MS/pp_step10_census.py "dp1 vs pp8 x vp4" $DUMP/dp1_A $DUMP/pp8_vp4
rm -rf $DUMP $OUT/ind_* $OUT/tri_*
echo "PPPROBE33 DONE $OUT"
