#!/bin/bash
# Where the CP forward first departs from dp1: the stream after the embedding and after every layer,
# first micro-batch of step 1, dp1 (256 tokens) against cp2 (rank r holds tokens [128r, 128r+128)).
# Tree /tmp/wt_cplayer = edc4cd71b + the registry alias + the layer-dump hack. Same seed and batch as
# the CP matrix; Attention Gym b19162e.
set -uo pipefail
T=/tmp/wt_cplayer; OUT=/workspace/cp_layer; mkdir -p $OUT
SEED=/workspace/.mx3_seeds_main/kimi_k3_debugmodel_f9365ce46c53/seed_ckpt
export TORCHINDUCTOR_CACHE_DIR=$OUT/inductor TRITON_CACHE_DIR=$OUT/triton PYTHONPATH=/tmp/attn_gym_up:$T
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
source /venv/main/bin/activate
run() { local nm=$1 np=$2 gpus=$3 cfg=$4; shift 4; local d=$OUT/run_$nm; rm -rf $d; mkdir -p $d; cp -r $SEED $d/checkpoint
  ( cd $T && CUDA_VISIBLE_DEVICES=$gpus LAYER_DUMP=$OUT/$nm timeout 1800 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
    --training.steps 1 $B --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 1 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  rm -rf $d/checkpoint; echo "$nm rc=$? $(ls $OUT/$nm.rank*.pt 2>/dev/null | wc -l) dumps"; }
run dp1 1 7 kimi_k3_debugmodel --parallelism.spmd_backend spmd_types &
run cp2 2 5,6 kimi_k3_debugmodel_cp2 &
wait
echo "CP LAYER PROBE DONE"
