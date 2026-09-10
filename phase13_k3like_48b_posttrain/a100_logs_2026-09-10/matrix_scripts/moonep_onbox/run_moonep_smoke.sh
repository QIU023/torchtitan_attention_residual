#!/bin/bash
# torchtitan: kimi_k3_debugmodel_moonep ep2 x fsdp2, 3 steps, on the k3_on_4025 worktree.
S=<scratch>; T=/workspace/tt_k3_on_4025
source /venv/main/bin/activate; cd $T
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH NCCL_NVLS_ENABLE=0
export MOONEP_MEM_HANDLE_TYPE=${MOONEP_MEM_HANDLE_TYPE:-auto}
rm -rf /workspace/smoke_moonep
PYTHONPATH=$T timeout 1500 torchrun --nproc_per_node=2 --master_port=41401 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_moonep \
  --debug.seed 42 --debug.deterministic --training.steps 3 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 --dump-folder /workspace/smoke_moonep
echo "SMOKE_EXIT=$?"
