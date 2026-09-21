#!/bin/bash
# Seeded matrix on the 09-22 rebuild (PP round 3 under the tree): the morning's 16 cells plus the five CP-composition
# cells (cp2 x tp2 both flavours, the three three-axis cells), one seed checkpoint, one warm inductor cache.
set -uo pipefail
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
RB=torchtitan_recipes.tests.b200
D1="--parallelism.data_parallel_shard_degree 1"; D2="--parallelism.data_parallel_shard_degree 2"
PP2="--parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
TP2="--parallelism.tensor_parallel_degree 2"; EP2="--parallelism.expert_parallel_degree 2"
cd /workspace && TITAN=/tmp/wt_int0922 VENV=/workspace/venv_bfx9 PYPRE=$S:/workspace/pylib/attn_gym_main CFG=kimi_k3_debugmodel \
  BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256" \
  CELLS="dp1|1|kimi_k3|kimi_k3_debugmodel|$D1
fsdp2|2|kimi_k3|kimi_k3_debugmodel|$D2
tp2|2|kimi_k3|kimi_k3_debugmodel|$D1 $TP2
ep2_fsdp2|2|kimi_k3|kimi_k3_debugmodel|$D2 $EP2
tp2_ep2|2|kimi_k3|kimi_k3_debugmodel|$D1 $TP2 $EP2
pp2|2|kimi_k3|kimi_k3_debugmodel|$D1 $PP2 --parallelism.num_pp_microbatches 16
fsdp2_pp2|4|kimi_k3|kimi_k3_debugmodel|$D2 $PP2 --parallelism.num_pp_microbatches 8
tp2_pp2|4|kimi_k3|kimi_k3_debugmodel|$D1 $TP2 $PP2 --parallelism.num_pp_microbatches 16
fsdp2_tp2_pp2|8|kimi_k3|kimi_k3_debugmodel|$D2 $TP2 $PP2 --parallelism.num_pp_microbatches 8
fsdp2_pp2_ep2|4|kimi_k3|kimi_k3_debugmodel|$D2 $EP2 $PP2 --parallelism.num_pp_microbatches 8
fsdp2_tp2_ep2|4|$RB|kimi_k3_debugmodel_mm|
cp2_ag|2|$RB|kimi_k3_debugmodel_mm_allgather_kv_cp2|
cp2_ul|2|$RB|kimi_k3_debugmodel_mm_ulysses_cp2|
cp2_ep2|2|$RB|kimi_k3_debugmodel_mm_allgather_kv_cp2|$EP2
cp2_fsdp2|4|$RB|kimi_k3_debugmodel_mm_allgather_kv_cp2|$D2
ep8_fsdp8|8|kimi_k3|kimi_k3_debugmodel|--parallelism.data_parallel_shard_degree 8 --parallelism.expert_parallel_degree 8
cp2_tp2_ag|4|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_tp2_min96|
cp2_tp2_ul|4|cpmm_probe|kimi_k3_mm_ulysses_cp2_tp2_min96|
fsdp2_cp2_tp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_tp2_min96|$D2
fsdp2_cp2_pp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_pp2_min96|$D2 --parallelism.num_pp_microbatches 8
tp2_cp2_pp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_pp2_min96|$TP2 --parallelism.no-enable-sequence-parallel --parallelism.num_pp_microbatches 16" $M/mx4.sh int0922
echo "MATRIX DONE"
