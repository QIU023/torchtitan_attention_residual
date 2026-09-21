#!/bin/bash
# Part B (tp cells carry ep2: main's #4794 refuses ep < tp on MoE models): the four 8-GPU cells, after part A and the 4-GPU veRL cells.
# Seeded matrix on the 09-22 rebuild rebased onto 63c4e9fef (#4808 stacked w13): the same 21 cells through mx4b.sh (checkpointer via the mx4ckpt wrapper).
# cells (cp2 x tp2 both flavours, the three three-axis cells), one seed checkpoint, one warm inductor cache.
set -uo pipefail
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
M=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
RB=torchtitan_recipes.tests.b200
D1="--parallelism.data_parallel_shard_degree 1"; D2="--parallelism.data_parallel_shard_degree 2"
PP2="--parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
TP2="--parallelism.tensor_parallel_degree 2"; EP2="--parallelism.expert_parallel_degree 2"
cd /workspace && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TITAN=/tmp/wt_int0922b SEED_ROOT=/workspace/.mx4_seeds_0922b VENV=/workspace/venv_bfx9 PYPRE=$S:/workspace/pylib/attn_gym_main CFG=kimi_k3_debugmodel \
  BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256" \
  CELLS="fsdp2_tp2_pp2|8|kimi_k3|kimi_k3_debugmodel|$D2 $TP2 $PP2 --parallelism.num_pp_microbatches 8 $EP2
ep8_fsdp8|8|kimi_k3|kimi_k3_debugmodel|--parallelism.data_parallel_shard_degree 8 --parallelism.expert_parallel_degree 8
fsdp2_cp2_tp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_tp2_min96|$D2 $EP2
fsdp2_cp2_pp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_pp2_min96|$D2 --parallelism.num_pp_microbatches 8
tp2_cp2_pp2|8|cpmm_probe|kimi_k3_mm_allgather_kv_cp2_pp2_min96|$TP2 --parallelism.no-enable-sequence-parallel --parallelism.num_pp_microbatches 16 $EP2" $M/mx4b.sh int0922b
echo "MATRIX DONE"
