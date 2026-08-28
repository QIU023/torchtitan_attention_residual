#!/bin/bash
# Is the standard-vs-minimal_async_ep gradient gap K3-specific? Same two backends on upstream
# deepseek_v3_debugmodel (ep2 x fsdp2, full AC on both, deterministic, seed 42, 2 steps).
# Also: MinimalAsyncEP's own GPU kernel unit tests on this torch/triton.
set -u
T=/workspace/tt_ep_review1; S=<scratch>
source /venv/main/bin/activate; cd $T
echo "### KERNEL_UNIT_TESTS (skipped on rerun)"; false && \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$T timeout 900 python -m pytest tests/unit_tests/gpu/test_minimal_async_ep_kernels.py -q 2>&1 | tail -3
COMMON="--debug.seed 42 --debug.deterministic --training.steps 2 --metrics.log_freq 1 --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 --training.disable_cuda_graphs"
for cfg in deepseek_v3_debugmodel deepseek_v3_debugmodel_minimal_async_ep; do
  echo "### DSV3 $cfg"
  rm -rf /workspace/dsv3_$cfg
  NCCL_NVLS_ENABLE=0 PYTHONPATH=$T timeout 1200 torchrun --nproc_per_node=2 --master_port=41100 -m torchtitan.train --module deepseek_v3 --config $cfg $COMMON --dump-folder /workspace/dsv3_$cfg activation-checkpoint:full > $S/dsv3_$cfg.log 2>&1
  echo "rc=$?"; grep -oE "step: +[0-9]+ .*loss: *[0-9.]+ .*grad_norm: *[0-9.]+" $S/dsv3_$cfg.log | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "step: +[0-9]+|loss: *[0-9.]+|grad_norm: *[0-9.]+" | paste - - - | sort -u
done
echo "### DISCRIMINATOR_DONE"
