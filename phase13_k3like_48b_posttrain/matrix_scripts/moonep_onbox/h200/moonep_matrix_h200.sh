#!/bin/bash
# std vs moonep cells on the K3 debug model, one shared warm inductor cache per invocation.
# Usage: moonep_matrix_h200.sh <tag>   (cells with nproc 2 and 4)
source /workspace/kit/h200/env.sh
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
export BATCH="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
export CELLS="dp2_std|2|kimi_k3_debugmodel||$D 2
dp2ep2_std|2|kimi_k3_debugmodel||$D 2 $E 2
dp2ep2_moonep|2|kimi_k3_debugmodel_moonep||$D 2 $E 2
dp2ep2_std_again|2|kimi_k3_debugmodel||$D 2 $E 2
dp4ep4_std|4|kimi_k3_debugmodel||$D 4 $E 4
dp4ep4_moonep|4|kimi_k3_debugmodel_moonep||$D 4 $E 4"
exec bash /workspace/kit/h200/mx3_backend_h200.sh "$1"
