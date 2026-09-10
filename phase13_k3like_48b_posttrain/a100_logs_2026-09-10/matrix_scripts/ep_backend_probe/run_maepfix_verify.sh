#!/bin/bash
# Verify the MinimalAsyncEP owned-dispatch fix: (1) deepseek plain probe on the fix branch,
# (2) K3 probe on ep_review1 + fix, (3) K3 ep2 x minimal_async_ep 10-step cell vs standard.
S=<scratch>
source /venv/main/bin/activate
export NCCL_NVLS_ENABLE=0
echo "### DSV3_FIX"
T=/workspace/tt_maep_fix; cd $T
for cfg in dsv3_std dsv3_maep_plain; do
  D=/workspace/gpf_$cfg; rm -rf $D; mkdir -p $D
  GRAD_PROBE_OUT=$S/grads_fix_$cfg.json PYTHONPATH=$T:$S timeout 1200 torchrun --nproc_per_node=2 --master_port=41600 $S/grad_probe.py \
    --module dsv3_probe --config $cfg --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
    --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 --dump-folder $D > $S/gpf_$cfg.log 2>&1
  echo "$cfg rc=$?"; grep -h "GRAD_PROBE_OK" $S/gpf_$cfg.log | head -1 | cut -c1-160
done
python $S/grad_diff.py $S/grads_fix_dsv3_std.json $S/grads_fix_dsv3_maep_plain.json 2>&1 | head -14
echo "### K3_FIX_PROBE"
T=/workspace/tt_ep_review1; cd $T; SEED=/workspace/mx3_ep_backends_0828_0828_145737/seed/checkpoint
for cfg in kimi_k3_debugmodel kimi_k3_debugmodel_minimal_async_ep; do
  D=/workspace/gpf_$cfg; rm -rf $D; mkdir -p $D; cp -r $SEED $D/checkpoint
  GRAD_PROBE_OUT=$S/grads_fix_$cfg.json PYTHONPATH=$T timeout 1500 torchrun --nproc_per_node=2 --master_port=41601 $S/grad_probe.py \
    --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
    --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
    --dump-folder $D activation-checkpoint:full > $S/gpf_$cfg.log 2>&1
  echo "$cfg rc=$?"; grep -h "GRAD_PROBE_OK" $S/gpf_$cfg.log | head -1 | cut -c1-160; rm -rf $D/checkpoint
done
python $S/grad_diff.py $S/grads_fix_kimi_k3_debugmodel.json $S/grads_fix_kimi_k3_debugmodel_minimal_async_ep.json 2>&1 | head -16
echo "### K3_FIX_CELL"
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
export TITAN=/workspace/tt_ep_review1 BATCH="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CELLS="ep2_standard|2|kimi_k3_debugmodel|NCCL_NVLS_ENABLE=0|$D 2 $E 2
ep2_minimal_async_ep|2|kimi_k3_debugmodel_minimal_async_ep|NCCL_NVLS_ENABLE=0|$D 2 $E 2"
$M/mx3_backend.sh maepfix_0828 2>&1 | tail -4
echo "### VERIFY_DONE"
