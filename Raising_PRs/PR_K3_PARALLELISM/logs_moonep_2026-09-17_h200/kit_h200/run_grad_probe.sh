#!/bin/bash
# One micro-batch forward+backward from the seed checkpoint, per-parameter grad norms:
# std and moonep on one shared inductor cache, plus std again on a fresh cache as the floor row.
# Usage: SEED=<dir with checkpoint/> NP=2 run_grad_probe.sh
source /workspace/kit/h200/env.sh; NP=${NP:-2}; SEED=${SEED:?seed dir}; S=/workspace/results; cd $TITAN
SHARED=/workspace/gp_cache_np$NP; FRESH=/workspace/gp_cache_fresh_np${NP}_$(date +%H%M%S); rm -rf $SHARED $FRESH
run_one(){ local tag=$1 cfg=$2 cache=$3
  D=/workspace/gp_${tag}_np$NP; rm -rf $D; mkdir -p $D; cp -r $SEED/checkpoint $D/checkpoint
  echo "### PROBE $tag ($cfg) NP=$NP cache=$(basename $cache)"
  TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton GRAD_PROBE_OUT=$S/grads_${tag}_np$NP.json timeout 1500 torchrun --nproc_per_node=$NP --master_port=$((41500+NP)) /workspace/kit/h200/grad_probe.py \
    --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
    --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree $NP --parallelism.expert_parallel_degree $NP \
    --dump-folder $D > $S/gp_${tag}_np$NP.log 2>&1
  echo "rc=$?"; grep -h "GRAD_PROBE_OK" $S/gp_${tag}_np$NP.log | head -1 | cut -c1-160; grep -h -m1 "Error\|error:" $S/gp_${tag}_np$NP.log | grep -v lspci | cut -c1-200
  rm -rf $D/checkpoint; }
run_one std kimi_k3_debugmodel $SHARED
run_one moonep kimi_k3_debugmodel_moonep $SHARED
run_one std_fresh kimi_k3_debugmodel $FRESH
echo "### PROBE_DONE NP=$NP"
echo "### grad diff std vs moonep (shared cache)"; python /workspace/kit/grad_diff.py $S/grads_std_np$NP.json $S/grads_moonep_np$NP.json 2>&1 | tee $S/grad_diff_np$NP.txt
echo "### grad diff std vs std_fresh (floor)"; python /workspace/kit/grad_diff.py $S/grads_std_np$NP.json $S/grads_std_fresh_np$NP.json 2>&1 | tee $S/grad_diff_floor_np$NP.txt
