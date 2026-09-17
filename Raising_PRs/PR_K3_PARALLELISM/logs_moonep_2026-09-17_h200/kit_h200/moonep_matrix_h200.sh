#!/bin/bash
# One invocation per table (one warm inductor cache per table): dp2, dp4, and the fresh-cache
# floor rows dp2floor / dp4floor (the standard ep cell alone on its own cache).
# Usage: moonep_matrix_h200.sh <dp2|dp2floor|dp4|dp4floor>
source /workspace/kit/h200/env.sh
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
export BATCH="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
case "$1" in
  dp2) export CELLS="dp2_std|2|kimi_k3_debugmodel||$D 2
dp2_moonep_ep1|2|kimi_k3_debugmodel_moonep||$D 2
dp2ep2_std|2|kimi_k3_debugmodel||$D 2 $E 2
dp2ep2_moonep|2|kimi_k3_debugmodel_moonep||$D 2 $E 2";;
  dp2floor) export CELLS="dp2ep2_std_fresh|2|kimi_k3_debugmodel||$D 2 $E 2";;
  dp4) export CELLS="dp4_std|4|kimi_k3_debugmodel||$D 4
dp4ep4_std|4|kimi_k3_debugmodel||$D 4 $E 4
dp4ep4_moonep|4|kimi_k3_debugmodel_moonep||$D 4 $E 4";;
  dp4floor) export CELLS="dp4ep4_std_fresh|4|kimi_k3_debugmodel||$D 4 $E 4";;
  c4dp2) export STEPS=100; export CELLS="dp2_std_c4|2|kimi_k3_debugmodel_c4||$D 2
dp2ep2_std_c4|2|kimi_k3_debugmodel_c4||$D 2 $E 2
dp2ep2_moonep_c4|2|kimi_k3_debugmodel_moonep_c4||$D 2 $E 2";;
  c4dp2floor) export STEPS=100; export CELLS="dp2ep2_std_c4_fresh|2|kimi_k3_debugmodel_c4||$D 2 $E 2";;
  c4dp4) export STEPS=100; export CELLS="dp4_std_c4|4|kimi_k3_debugmodel_c4||$D 4
dp4ep4_std_c4|4|kimi_k3_debugmodel_c4||$D 4 $E 4
dp4ep4_moonep_c4|4|kimi_k3_debugmodel_moonep_c4||$D 4 $E 4";;
  c4dp4floor) export STEPS=100; export CELLS="dp4ep4_std_c4_fresh|4|kimi_k3_debugmodel_c4||$D 4 $E 4";;
  *) echo "family?"; exit 2;;
esac
SEED_CFG=$( [[ "$1" == c4* ]] && echo kimi_k3_debugmodel_c4 || echo kimi_k3_debugmodel ) exec bash /workspace/kit/h200/mx3_backend_h200.sh "moonep_$1"
