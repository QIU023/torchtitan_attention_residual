#!/bin/bash
# MoonEP vs standard on the k3_on_4025 tree: ep2 x fsdp2 (MoonEP needs dp_shard == ep), 2 GPUs,
# same seed checkpoint, steps 1 / 3 / 10. Usage: TITAN=<k3_on_4025 worktree> moonep_matrix.sh <tag>
set -u
export TITAN
MOON_ENV="CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} NCCL_NVLS_ENABLE=0 MOONEP_MEM_HANDLE_TYPE=${MOONEP_MEM_HANDLE_TYPE:-auto}"
COMMON_ENV="NCCL_NVLS_ENABLE=0"
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
export BATCH="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CELLS="dp2|2|kimi_k3_debugmodel|$COMMON_ENV|$D 2
ep2_standard|2|kimi_k3_debugmodel|$COMMON_ENV|$D 2 $E 2
ep2_moonep|2|kimi_k3_debugmodel_moonep|$MOON_ENV|$D 2 $E 2"
exec "$(dirname "$0")/mx3_backend.sh" "$1"
