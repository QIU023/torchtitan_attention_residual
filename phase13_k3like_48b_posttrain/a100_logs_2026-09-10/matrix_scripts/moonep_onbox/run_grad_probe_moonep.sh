#!/bin/bash
# Per-parameter gradient probe, k3_on_4025 tree: standard vs moonep, ep2 x fsdp2, one microbatch, same seed ckpt.
S=<scratch>; T=/workspace/tt_k3_on_4025
SEED=${SEED:?path to seed checkpoint dir}
source /venv/main/bin/activate; cd $T
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH NCCL_NVLS_ENABLE=0 MOONEP_MEM_HANDLE_TYPE=${MOONEP_MEM_HANDLE_TYPE:-auto}
for cfg in kimi_k3_debugmodel kimi_k3_debugmodel_moonep; do
  D=/workspace/gpm_$cfg; rm -rf $D; mkdir -p $D; cp -r $SEED $D/checkpoint
  echo "### PROBE $cfg"
  GRAD_PROBE_OUT=$S/grads_m_$cfg.json PYTHONPATH=$T timeout 1500 torchrun --nproc_per_node=2 --master_port=41500 $S/grad_probe.py \
    --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
    --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
    --dump-folder $D > $S/gpm_$cfg.log 2>&1
  echo "rc=$?"; grep -h "GRAD_PROBE_OK" $S/gpm_$cfg.log | head -1; grep -h -m1 "Error\|error:" $S/gpm_$cfg.log | cut -c1-200
  rm -rf $D/checkpoint
done
echo "### PROBE_DONE"
python $S/grad_diff.py $S/grads_m_kimi_k3_debugmodel.json $S/grads_m_kimi_k3_debugmodel_moonep.json 2>&1
