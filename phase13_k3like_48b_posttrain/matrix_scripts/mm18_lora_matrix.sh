#!/bin/bash
# The mm_lora arm: the same 18 cells on the LoRA flavor (kimi_k3_debugmodel_lora, core LoRAConverter on the
# attention, FFN and MoE projections) on mm18_lora_int = mm18_int + lora_review1 (edaa536e3); the CP cells
# through a local kimi_k3_debugmodel_lora_cp2 alias. Derived from mm18_matrix.sh by sed on 2026-09-06.
# The 18-cell multimodal parallelism matrix (MATRIX_18_CORRECTED_2026-08-09.md's cells) on the
# new tree: TP/SP + CP (Attention Gym recipe) + rebased PP on one scratch integration branch
# (mm18_int, /tmp/wt_int18). kimi_k3_debugmodel at 33 layers, spmd_types everywhere, 4096 tokens
# per step in 256-token micro-batches (the PP tables' batch), 8 pipeline micro-batches, SP on
# under TP except where CP is also on (the vision splice under CP refuses SP).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel_lora PYPRE=/tmp/attn_gym_up
export TITAN=/tmp/wt_int18l
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
P="--parallelism.pipeline_parallel_degree"; C="--parallelism.context_parallel_degree"
E="--parallelism.expert_parallel_degree"; MB="--parallelism.num-pp-microbatches 8"
S="--parallelism.spmd_backend spmd_types"; NOSP="--parallelism.no-enable-sequence-parallel"
export BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
PLAIN=kimi_k3_debugmodel_lora; CPF=kimi_k3_debugmodel_lora_cp2

streamA() {  # GPUs 0,1
  export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_mm18l_a
  CFG=$PLAIN CELLS="dp1|1|$D 1 $S
fsdp2|2|$D 2 $S
pp2|2|$D 1 $P 2 $MB $S
tp2|2|$D 1 $T 2 $S
ep2_fsdp2|2|$D 2 $E 2 $S" $MX mm18l_a2
  CFG=$CPF CELLS="cp2|2|$D 1" $MX mm18l_a2cp
}
streamB() {  # GPUs 2-5
  export CUDA_VISIBLE_DEVICES=2,3,4,5 TRITON_CACHE_DIR=/workspace/.triton_mm18l_b
  CFG=$PLAIN CELLS="pp4|4|$D 1 $P 4 $MB $S
tp4|4|$D 1 $T 4 $S" $MX mm18l_b4
  CFG=$CPF CELLS="cp4|4|$D 1 $C 4" $MX mm18l_b4cp
}
streamC() {  # all 8 GPUs, after A and B
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRITON_CACHE_DIR=/workspace/.triton_mm18l_c
  CFG=$PLAIN CELLS="pp8|8|$D 1 $P 8 $MB $S
ep8_fsdp8|8|$D 8 $E 8 $S
fsdp2_tp2_pp2|8|$D 2 $T 2 $P 2 $MB $S
ep2_fsdp2_tp2_pp2|8|$D 2 $E 2 $T 2 $P 2 $MB $S" $MX mm18l_c8
  CFG=$CPF CELLS="fsdp2_pp2_cp2|8|$D 2 $P 2 $MB
ep2_fsdp2_pp2_cp2|8|$D 2 $E 2 $P 2 $MB
fsdp2_tp2_cp2|8|$D 2 $T 2 $NOSP
tp2_pp2_cp2|8|$D 1 $T 2 $P 2 $MB $NOSP
ep2_fsdp2_tp2_cp2|8|$D 2 $E 2 $T 2 $NOSP" $MX mm18l_c8cp
}
streamA & streamB & wait
streamC
echo "MM18 LORA MATRIX DONE"
