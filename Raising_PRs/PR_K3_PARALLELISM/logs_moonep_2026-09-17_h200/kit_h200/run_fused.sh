#!/bin/bash
# The fused-transport head: row probe, hot probe, self-tests on cutlass 4.6.0 + the make_rmem_tensor patch, then the tables.
source /workspace/kit/h200/env.sh; R=/workspace/results/summary.txt; mkdir -p /workspace/results
echo "== $(date -u +%FT%TZ) FUSED head $(git -C $TITAN rev-parse --short HEAD), cutlass-dsl 4.6.0 + moonep_grad_reduce_cutlass46.patch" | tee -a $R
echo "== $(date -u +%FT%TZ) row occupancy probe dp2 x ep2, 1 step" | tee -a $R
cd $TITAN; timeout 900 torchrun --nproc_per_node=2 --master_port=41850 /workspace/kit/h200/moonep_row_occupancy_probe.py --module kimi_k3 --config kimi_k3_debugmodel_moonep --debug.seed 42 --debug.deterministic --training.steps 1 --metrics.log_freq 1 --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 --dump-folder /workspace/rowprobe > /workspace/results/row_probe_np2_fused.log 2>&1; rc=$?
echo "row probe rc=$rc OK=$(grep -ac "OK only own" /workspace/results/row_probe_np2_fused.log) VIOLATION=$(grep -ac VIOLATION /workspace/results/row_probe_np2_fused.log) $(sed "s/\x1b\[[0-9;]*m//g" /workspace/results/row_probe_np2_fused.log | grep -aoE "step: *1 .*grad_norm: *[0-9.]+" | head -1 | cut -c1-70)" | tee -a $R
echo "== $(date -u +%FT%TZ) hot probe np2" | tee -a $R; NP=2 bash /workspace/kit/h200/run_hot_probe.sh 2>&1 | tee -a $R
echo "== $(date -u +%FT%TZ) selftests np2 (4.6.0 + patch)" | tee -a $R; NP=2 bash /workspace/kit/h200/run_selftests.sh 2>&1 | tee -a $R
bash /workspace/kit/h200/run_rest.sh
