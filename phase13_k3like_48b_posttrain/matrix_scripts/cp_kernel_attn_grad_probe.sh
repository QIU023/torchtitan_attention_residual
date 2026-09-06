#!/bin/bash
# Where the packed / generic Ulysses gradients part: the MLA inner kernel's backward at layer 23 (the
# first MLA layer the backward meets; everything downstream is kernel-free, so grad_output must be
# bitwise) and layer 3. First micro-batch of step 1, both CP ranks, on /tmp/wt_cpdump.
set -uo pipefail
T=/tmp/wt_cpdump; OUT=/workspace/cp_kernel_dump; mkdir -p $OUT
SEED=/workspace/.mx3_seeds_main/kimi_k3_debugmodel_f9365ce46c53/seed_ckpt
export CUDA_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_CACHE_DIR=$OUT/inductor TRITON_CACHE_DIR=$OUT/triton PYTHONPATH=/tmp/attn_gym_up:$T
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
source /venv/main/bin/activate
run() { local nm=$1 cfg=$2; local d=$OUT/run_ag_$nm; rm -rf $d; mkdir -p $d; cp -r $SEED $d/checkpoint
  ( cd $T && ATTN_GRAD_DUMP=$OUT/attn_$nm GRAD_TENSOR_DUMP=$OUT/scratch_$nm timeout 1800 torchrun --nproc_per_node=2 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train \
    --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 1 $B \
    --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 1 --dump-folder $d > $OUT/attn_$nm.log 2>&1 )
  rm -rf $d/checkpoint $OUT/scratch_$nm.rank0.pt; echo "$nm rc=$? $(ls $OUT/attn_$nm.L*.pt 2>/dev/null | wc -l) hook dumps"; }
run packed_ulysses kimi_k3_debugmodel_cp2
run generic_ulysses kimi_k3_debugmodel_cp2_generic
echo "ATTN GRAD PROBE DONE"
