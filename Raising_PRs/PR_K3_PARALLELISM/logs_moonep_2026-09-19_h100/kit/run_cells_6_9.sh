#!/bin/bash
# Cell 9 (row occupancy, the precondition for trusting cell 3) and cell 6
# (planning under a slot count below what the routing needs).
source /workspace/kit/h200/env.sh
R=/workspace/results/summary.txt
cd "$TITAN"
COMMON="--module kimi_k3 --debug.seed 42 --debug.deterministic --training.steps 1"
BATCH="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"

echo "== $(date -u +%FT%TZ) cell 9 row occupancy, ep4" | tee -a $R
timeout 1200 torchrun --nproc_per_node=4 --master_port=41901 \
  /workspace/kit/h200/moonep_row_occupancy_probe.py $COMMON \
  --config kimi_k3_debugmodel_moonep $BATCH \
  --parallelism.expert_parallel_degree 4 --parallelism.data_parallel_shard_degree 4 \
  > /workspace/results/cell9_occupancy.log 2>&1
echo "cell 9 rc=$?" | tee -a $R
grep -E "VIOLATION|checks|OCCUPANCY" /workspace/results/cell9_occupancy.log | tail -6 | tee -a $R

for B in b7 b1; do
  echo "== $(date -u +%FT%TZ) cell 6 slots=$B, forced hot, ep4" | tee -a $R
  MOONEP_FORCE_HOT=1 timeout 900 torchrun --nproc_per_node=4 --master_port=4191${B: -1} \
    -m torchtitan.train $COMMON --config kimi_k3_debugmodel_moonep_$B $BATCH \
    --parallelism.expert_parallel_degree 4 --parallelism.data_parallel_shard_degree 4 \
    > /workspace/results/cell6_$B.log 2>&1
  rc=$?
  echo "cell 6 $B rc=$rc" | tee -a $R
  # the plan asks how it fails: an error naming the bound, not a hang
  grep -oiE "slot|capacity|prefetch|exceed|not enough|bound|assert[^\"]{0,60}" \
    /workspace/results/cell6_$B.log | sort -u | head -5 | tee -a $R
done
echo "== $(date -u +%FT%TZ) CELLS_6_9_DONE" | tee -a $R
