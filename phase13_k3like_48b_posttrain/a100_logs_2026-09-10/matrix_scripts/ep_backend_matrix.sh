#!/bin/bash
# EP comm-backend matrix for the K3 EP PR (review ask: "did you try other backends?").
# 2 GPUs: dp2 baseline, then ep2 x fsdp2 on standard / deepep / minimal_async_ep.
# minimal_async_ep requires full AC, so a standard + full-AC control cell isolates the AC effect.
# hybridep is GB200/NVL72-only (its own DeepEP branch) and is not run here.
# Usage: TITAN=<ep_review1 worktree> ep_backend_matrix.sh <tag>
set -u
export TITAN
SP=/venv/main/lib/python3.12/site-packages
# DeepEP v2 on a host without an RDMA NIC (torchtitan qwen3_moe_deepep docstring).
DEEPEP_ENV="CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} NCCL_NVLS_ENABLE=0 EP_DISABLE_GIN=1 EP_REUSE_NCCL_COMM=0 NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_DISABLE_MNNVL=1 LD_LIBRARY_PATH=$SP/nvidia/nvshmem/lib:$SP/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
COMMON_ENV="NCCL_NVLS_ENABLE=0"
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
export BATCH="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CELLS="dp2|2|kimi_k3_debugmodel|$COMMON_ENV|$D 2
ep2_standard|2|kimi_k3_debugmodel|$COMMON_ENV|$D 2 $E 2
ep2_standard_fullac|2|kimi_k3_debugmodel|$COMMON_ENV|$D 2 $E 2 activation-checkpoint:full
ep2_minimal_async_ep|2|kimi_k3_debugmodel_minimal_async_ep|$COMMON_ENV|$D 2 $E 2
ep2_deepep|2|kimi_k3_debugmodel_deepep|$DEEPEP_ENV|$D 2 $E 2"
exec "$(dirname "$0")/mx3_backend.sh" "$1"
