#!/bin/bash
# CP x TP x PP 3D verification matrix (2026-07-24, 8x5060Ti).
# Reproduces CP_TP_3D_VERIFICATION_2026-07-24.md. Debug model, seed 42
# deterministic; parity cells use 20 steps, composition cells 5-10.
# PASS = descending finite loss; parity cells additionally compared
# against their 1-axis baselines (see the report for bands).
set -u
cd /workspace/torchtitan_attention_residual/torchtitan
CFG="--module kimi_k3 --config kimi_linear_debugmodel --checkpoint.no-enable --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
PORT=29500
run_cell() {
  local name="$1" ngpu="$2" steps="$3"; shift 3
  PORT=$((PORT+1))
  echo "=================== CELL: $name ==================="
  local out clean
  out=$(CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) torchrun --nproc_per_node=$ngpu \
        --master_port=$PORT -m torchtitan.train $CFG --training.steps $steps "$@" 2>&1)
  clean=$(echo "$out" | sed -E 's/\x1b\[[0-9;]*m//g')
  echo "$clean" | grep -E "step: +[0-9]+ +loss" \
    | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+  grad_norm: +[-0-9.]+).*/\1/' \
    | grep -vE '\-4\.00000|\-2\.00000' | sort -u | sed -n '1p;$p'
  echo "$clean" | grep -iE "traceback|RuntimeError|ValueError|NotImplementedError" | head -2
}

# --- parity legs (compare against each other; see report) ---
run_cell "cp1 baseline"       1 20
run_cell "cp2"                2 20 --parallelism.data_parallel_shard_degree 1 --parallelism.context_parallel_degree 2
run_cell "cp4"                4 20 --parallelism.data_parallel_shard_degree 1 --parallelism.context_parallel_degree 4
run_cell "tp2"                2 20 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2
run_cell "tp2cp2"             4 20 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2

# --- the 3D target + composition cells ---
run_cell "tp2cp2pp2 (3D)"     8 10 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B
run_cell "fsdp2tp2cp2"        8 10 --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2
run_cell "tp2cp2ep2"          8 10 --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2 --parallelism.expert_parallel_degree 2
run_cell "cp2fsdp2"           4  5 --parallelism.data_parallel_shard_degree 2 --parallelism.context_parallel_degree 2
run_cell "cp2ep2"             8  5 --parallelism.data_parallel_shard_degree 4 --parallelism.context_parallel_degree 2 --parallelism.expert_parallel_degree 2
run_cell "cp2pp2fsdp2"        8  5 --parallelism.data_parallel_shard_degree 2 --parallelism.context_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B

# --- regression cells (non-CP paths must stay green) ---
run_cell "fsdp8"              8  5 --parallelism.data_parallel_shard_degree 8
run_cell "pp2fsdp4"           8  5 --parallelism.data_parallel_shard_degree 4 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B
run_cell "4d fsdp2tp2pp2ep2"  8  5 --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B --parallelism.expert_parallel_degree 2

echo "=================== MATRIX DONE ==================="
# LoRA / gated flavors (separate configs):
#   LoRA x FSDP x TP x CP:
#     --config kimi_linear_debugmodel_gated_lora dp2 tp2 cp2   (PASS)
#   gated (k3faithful) x TP:
#     --config kimi_linear_debugmodel_k3faithful dp1 tp2       (PASS, was broken pre-fix)
# Known limitation: the LoRA flavor without FSDP (dp_shard=1) hits a
# preexisting fp32-vs-bf16 dtype crash in block_attn_res (not CP-related;
# repros at plain dp1 tp2 on the pre-CP commit 7d8acabe).
