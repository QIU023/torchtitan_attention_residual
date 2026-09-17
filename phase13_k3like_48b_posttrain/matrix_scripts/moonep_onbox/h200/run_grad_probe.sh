#!/bin/bash
# One micro-batch forward+backward from the seed checkpoint, per-parameter grad norms, std vs moonep.
# Usage: SEED=<dir with checkpoint/> NP=2 run_grad_probe.sh
source /workspace/kit/h200/env.sh; NP=${NP:-2}; SEED=${SEED:?seed dir}; S=/workspace/results; cd $TITAN
for cfg in kimi_k3_debugmodel kimi_k3_debugmodel_moonep; do
  D=/workspace/gp_${cfg}_np$NP; rm -rf $D; mkdir -p $D; cp -r $SEED/checkpoint $D/checkpoint
  echo "### PROBE $cfg NP=$NP"
  GRAD_PROBE_OUT=$S/grads_${cfg}_np$NP.json timeout 1500 torchrun --nproc_per_node=$NP --master_port=$((41500+NP)) /workspace/kit/grad_probe.py \
    --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
    --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree $NP --parallelism.expert_parallel_degree $NP \
    --dump-folder $D > $S/gp_${cfg}_np$NP.log 2>&1
  echo "rc=$?"; grep -h "GRAD_PROBE_OK" $S/gp_${cfg}_np$NP.log | head -1; grep -h -m1 "Error\|error:" $S/gp_${cfg}_np$NP.log | cut -c1-200
  rm -rf $D/checkpoint
done
echo "### PROBE_DONE NP=$NP"
python /workspace/kit/grad_diff.py $S/grads_kimi_k3_debugmodel_np$NP.json $S/grads_kimi_k3_debugmodel_moonep_np$NP.json 2>&1 | tee $S/grad_diff_np$NP.txt
