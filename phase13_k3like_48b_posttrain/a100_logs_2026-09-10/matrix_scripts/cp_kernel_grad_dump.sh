#!/bin/bash
# Step-1 gradients of cp2 under the four MLA kernels (packed / generic Ulysses, packed / generic
# all-gather KV), same seed and batch as the CP matrix, Attention Gym b19162e, on 61a73ca6c + the
# dump-and-exit hack. Forward is bitwise (the step-1 loss); this measures how far the backwards are.
set -uo pipefail
T=/tmp/wt_cpdump; OUT=/workspace/cp_kernel_dump; mkdir -p $OUT
SEED=/workspace/.mx3_seeds_main/kimi_k3_debugmodel_f9365ce46c53/seed_ckpt
export CUDA_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_CACHE_DIR=$OUT/inductor TRITON_CACHE_DIR=$OUT/triton PYTHONPATH=/tmp/attn_gym_up:$T
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
source /venv/main/bin/activate
run() { local nm=$1 cfg=$2; local d=$OUT/run_$nm; rm -rf $d; mkdir -p $d; cp -r $SEED $d/checkpoint
  ( cd $T && GRAD_TENSOR_DUMP=$OUT/$nm timeout 1800 torchrun --nproc_per_node=2 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train \
    --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 1 $B \
    --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 1 --dump-folder $d > $OUT/$nm.log 2>&1 )
  rm -rf $d/checkpoint; echo "$nm rc=$? $(ls -la $OUT/$nm.rank0.pt 2>/dev/null | awk '{print $5}')"; }
run packed_ulysses kimi_k3_debugmodel_cp2
run generic_ulysses kimi_k3_debugmodel_cp2_generic
run packed_ag kimi_k3_debugmodel_cp2_allgather
run generic_ag kimi_k3_debugmodel_cp2_allgather_generic
echo "CP KERNEL DUMP DONE"
