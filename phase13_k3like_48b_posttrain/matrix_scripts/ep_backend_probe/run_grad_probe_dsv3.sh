#!/bin/bash
S=<scratch>; T=/workspace/tt_ep_review1
source /venv/main/bin/activate; cd $T
for cfg in dsv3_std dsv3_maep_plain; do
  D=/workspace/gp_$cfg; rm -rf $D; mkdir -p $D
  echo "### PROBE $cfg"
  GRAD_PROBE_OUT=$S/grads_$cfg.json NCCL_NVLS_ENABLE=0 PYTHONPATH=$T:$S timeout 1200 torchrun --nproc_per_node=2 --master_port=41300 $S/grad_probe.py \
    --module dsv3_probe --config $cfg --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
    --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
    --dump-folder $D > $S/gp_$cfg.log 2>&1
  echo "rc=$?"; grep -h "GRAD_PROBE_OK" $S/gp_$cfg.log | head -1; grep -h -m1 "Error\|error:" $S/gp_$cfg.log | cut -c1-200
done
echo "### PROBE_DONE"
python $S/grad_diff.py $S/grads_dsv3_std.json $S/grads_dsv3_maep_plain.json 2>&1
