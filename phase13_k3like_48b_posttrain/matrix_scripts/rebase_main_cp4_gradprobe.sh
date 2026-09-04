#!/bin/bash
# Step-1 per-parameter gradients (fp32 norm + sha1) on cp_review4: dp1, the packed and the
# generic Ulysses, the packed and the generic all-gather. Same seed checkpoint, same batch,
# each config on its own warm inductor cache from the matrix run.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
T=/tmp/wt_cprun4; OUT=/workspace/gradprobe_cp4_full; rm -rf $OUT; mkdir -p $OUT
SEED=/workspace/.mx3_seeds_main/kimi_k3_debugmodel_f9365ce46c53/seed_ckpt
source /venv/main/bin/activate
cd $T && python $SP/grad_dump_hack_full.py $T && python $SP/grad_hash_hack.py $T
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
run(){ local nm=$1 np=$2 cfg=$3 cache=$4; shift 4
  local d=$OUT/$nm; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  CUDA_VISIBLE_DEVICES=4,5 TORCHINDUCTOR_CACHE_DIR=$cache GRAD_DUMP=$OUT/$nm PYTHONPATH=$T timeout 1500 torchrun --nproc_per_node=$np \
    --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module kimi_k3 --config $cfg \
    --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 1 $B \
    --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 1 "$@" \
    --dump-folder $d > $OUT/$nm.log 2>&1; echo "$nm rc=$? $(grep -oE 'step: *1 .*loss: *[0-9.]+' $OUT/$nm.log | grep -oE 'loss: *[0-9.]+')" >> $OUT/summary.txt
  rm -rf $d/checkpoint; }
run dp1 1 kimi_k3_debugmodel "$(ls -d /workspace/mx3_cp4_base_*/inductor | head -1)" --parallelism.spmd_backend spmd_types
run cp2 2 kimi_k3_debugmodel_cp2 "$(ls -d /workspace/mx3_cp4r_cp2_*/inductor | head -1)"
run cp2_generic 2 kimi_k3_debugmodel_cp2_generic "$(ls -d /workspace/mx3_cp4r_cp2_generic_*/inductor | head -1)"
run cp2_allgather 2 kimi_k3_debugmodel_cp2_allgather "$(ls -d /workspace/mx3_cp4r_cp2_allgather_*/inductor | head -1)"
run cp2_allgather_generic 2 kimi_k3_debugmodel_cp2_allgather_generic "$(ls -d /workspace/mx3_cp4r_cp2_allgather_generic_*/inductor | head -1)"
cd $T && git checkout -- torchtitan/trainer.py
echo "GRADPROBE DONE" >> $OUT/summary.txt
