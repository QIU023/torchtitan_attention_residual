#!/bin/bash
# Step-1 gradients of the same cell under both backends, on the declarations tree (dbc60701d plus the
# run worktree's local hacks and the dump-and-exit hack): dp2 and dp2 x ep2, partial_dtensor and
# spmd_types, from the matrix's seed checkpoint and batch. Each cell runs twice on one shared inductor
# cache; the second run's dump is the one compared (the first warms the cache; its dump is kept as the
# cold-vs-warm check). GPUs 2,3 -- the partial_dtensor matrix cells hold 0,1.
set -uo pipefail
T=/tmp/wt_decldump; OUT=/workspace/decl_dump; mkdir -p $OUT
SEED=/workspace/.mx3_seeds_main/kimi_k3_debugmodel_f9365ce46c53/seed_ckpt
export CUDA_VISIBLE_DEVICES=2,3 TORCHINDUCTOR_CACHE_DIR=$OUT/inductor TRITON_CACHE_DIR=$OUT/triton
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
S="--parallelism.spmd_backend spmd_types"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
source /venv/main/bin/activate
run() { local nm=$1 pass=$2; shift 2; local d=$OUT/run_${nm}_$pass; rm -rf $d; mkdir -p $d; cp -r $SEED $d/checkpoint
  ( cd $T && GRAD_TENSOR_DUMP=$OUT/${nm}_$pass PYTHONPATH=$T timeout 1800 torchrun --nproc_per_node=2 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
    --training.steps 1 $B --checkpoint.enable --checkpoint.interval 100000 "$@" --dump-folder $d > $OUT/${nm}_$pass.log 2>&1 )
  rm -rf $d/checkpoint; echo "$nm $pass rc=$? dump=$(ls -la $OUT/${nm}_$pass.rank0.pt 2>/dev/null | awk '{print $5}')"; }
for pass in warm dump; do
  run dp2_pd $pass $D 2 $PD
  run dp2_spmd $pass $D 2 $S
  run dp2_ep2_pd $pass $D 2 $E 2 $PD
  run dp2_ep2_spmd $pass $D 2 $E 2 $S
done
echo "DECL GRAD DUMP DONE"
