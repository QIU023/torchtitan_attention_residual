#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/compile_fix
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
C="--compile.enable --compile.components model"
run(){ local tree=$1 gpu=$2 name=$3 cache=$4 layers=$5 tiny=$6 steps=$7
  ( source /workspace/venv_bfx9/bin/activate && cd $S/$tree && CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=$cache/triton TORCHINDUCTOR_CACHE_DIR=$cache/inductor TORCHINDUCTOR_COMPILE_THREADS=8 TORCH_LOGS=recompiles \
    RP_LAYERS=$layers RP_TINY=$tiny PYTHONPATH=/workspace/pylib/attn_gym_main:$S/$tree timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel $B --training.steps $steps $C --dump-folder $S/out_$name > $S/$name.log 2>&1 ); echo "$name rc=$?" >> $S/status14.txt; rm -rf $S/out_$name; }
( run rp_notower 0 rp_notower_tiny33 $S/cache_rpn 33 1 2; run rp_notower 0 rp_notower_tiny93 $S/cache_rpn 93 1 2; run rp_notower 0 rp_notower_full24 $S/cache_w 24 0 3 ) &
( run rp_towerfirst 1 rp_towerfirst_tiny33 $S/cache_rpt 33 1 2; run rp_towerfirst 1 rp_towerfirst_tiny93 $S/cache_rpt 93 1 2; run rp_towerfirst 1 rp_towerfirst_full24 $S/cache_w 24 0 3 ) &
wait; echo ALL-DONE >> $S/status14.txt
