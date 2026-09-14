#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/compile_fix
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
C="--compile.enable --compile.components model"
I=$S/cache_final
run(){ local gpu=$1 name=$2 steps=$3; shift 3
  ( source /workspace/venv_bfx9/bin/activate && cd $S/final && CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=$I/triton TORCHINDUCTOR_CACHE_DIR=$I/inductor TORCHINDUCTOR_COMPILE_THREADS=8 \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$S/final timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel $B --training.steps $steps "$@" --dump-folder $S/out_$name > $S/$name.log 2>&1 ); echo "$name rc=$?" >> $S/status3.txt; rm -rf $S/out_$name; }
( run 0 f_warm 1 $C && run 0 f_compiled 3 $C ) &
( run 1 f_eager 3 && run 1 f_aot_eager 3 $C --compile.backend aot_eager ) &
wait; echo ALL-DONE >> $S/status3.txt
