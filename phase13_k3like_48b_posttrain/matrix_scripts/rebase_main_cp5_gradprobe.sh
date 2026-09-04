#!/bin/bash
# Step-1 cp-reduced full gradients on cp_review5: dp1, dp2 (the control), and cp2 with the
# four MLA kernels, each on its matrix cell's warm inductor cache, one shared seed.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
T=/tmp/wt_cprun5; OUT=/workspace/gradprobe_cp5; rm -rf $OUT; mkdir -p $OUT
SEED=$(ls -d /workspace/.mx3_seeds_main/kimi_k3_debugmodel_mm_*/seed_ckpt | head -1)
source /venv/main/bin/activate
cd $T && python $SP/grad_dump_hack_full.py $T && python $SP/grad_hash_hack.py $T
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
run(){ local nm=$1 np=$2 cfg=$3 cache=$4; shift 4
  local d=$OUT/$nm; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  TORCHINDUCTOR_CACHE_DIR=$cache GRAD_DUMP=$OUT/$nm PYTHONPATH=$T timeout 1500 torchrun --nproc_per_node=$np \
    --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module kimi_k3 --config $cfg \
    --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 1 $B \
    --checkpoint.enable --checkpoint.interval 100000 "$@" --dump-folder $d > $OUT/$nm.log 2>&1; echo "$nm rc=$? $(grep -oE 'step: *1 .*loss: *[0-9.]+' $OUT/$nm.log | grep -oE 'loss: *[0-9.]+')" >> $OUT/summary.txt
  rm -rf $d/checkpoint; }
BASE=$(ls -d /workspace/mx3_cp5_base_*/inductor | head -1)
run dp1 1 kimi_k3_debugmodel_mm "$BASE" --parallelism.data_parallel_shard_degree 1
run dp2 2 kimi_k3_debugmodel_mm "$BASE" --parallelism.data_parallel_shard_degree 2
for cfg in cp2 cp2_generic cp2_allgather cp2_allgather_generic; do
  run $cfg 2 kimi_k3_debugmodel_$cfg "$(ls -d /workspace/mx3_cp5_${cfg}_*/inductor | head -1)" --parallelism.data_parallel_shard_degree 1
done
cd $T && git checkout -- torchtitan/trainer.py
echo "GRADPROBE DONE" >> $OUT/summary.txt
