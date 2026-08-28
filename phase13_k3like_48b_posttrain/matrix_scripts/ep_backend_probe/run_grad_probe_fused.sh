#!/bin/bash
# Is the upstream fused_swiglu path affected too? standard+fused vs minimal_async_ep+fused, upstream main (no fix).
S=<scratch>; T=/workspace/tt_upstream_main
source /venv/main/bin/activate; cd $T; export NCCL_NVLS_ENABLE=0
for cfg in dsv3_std_fused dsv3_maep_fused; do
  D=/workspace/gpu_$cfg; rm -rf $D; mkdir -p $D
  GRAD_PROBE_OUT=$S/grads_fused_$cfg.json PYTHONPATH=$T:$S timeout 1500 torchrun --nproc_per_node=2 --master_port=41800 $S/grad_probe.py \
    --module dsv3_probe --config $cfg --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
    --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 --dump-folder $D > $S/gpu_$cfg.log 2>&1
  echo "$cfg rc=$?"; grep -h "GRAD_PROBE_OK" $S/gpu_$cfg.log | head -1 | cut -c1-160; grep -h -m1 "Error\|error:" $S/gpu_$cfg.log | cut -c1-200
done
echo "### FUSED_PROBE_DONE"
python $S/grad_diff.py $S/grads_fused_dsv3_std_fused.json $S/grads_fused_dsv3_maep_fused.json 2>&1 | head -16
