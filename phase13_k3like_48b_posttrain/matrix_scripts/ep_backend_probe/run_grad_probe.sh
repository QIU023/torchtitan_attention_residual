#!/bin/bash
S=<scratch>; T=/workspace/tt_ep_review1
SEED=/workspace/mx3_ep_backends_0828_0828_145737/seed/checkpoint
source /venv/main/bin/activate; cd $T
for cfg in kimi_k3_debugmodel kimi_k3_debugmodel_minimal_async_ep; do
  D=/workspace/gp_$cfg; rm -rf $D; mkdir -p $D; cp -r $SEED $D/checkpoint
  echo "### PROBE $cfg"
  GRAD_PROBE_OUT=$S/grads_$cfg.json NCCL_NVLS_ENABLE=0 PYTHONPATH=$T timeout 1200 torchrun --nproc_per_node=2 --master_port=41200 $S/grad_probe.py \
    --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
    --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
    --dump-folder $D activation-checkpoint:full > $S/gp_$cfg.log 2>&1
  echo "rc=$?"; grep -h "GRAD_PROBE_OK\|Loading the checkpoint from" $S/gp_$cfg.log | head -3
  rm -rf $D/checkpoint
done
echo "### PROBE_DONE"
python $S/grad_diff.py $S/grads_kimi_k3_debugmodel.json $S/grads_kimi_k3_debugmodel_minimal_async_ep.json 2>&1
