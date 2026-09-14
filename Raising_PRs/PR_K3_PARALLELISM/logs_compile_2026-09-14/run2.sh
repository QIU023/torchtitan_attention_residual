#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/compile_fix
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
C="--compile.enable --compile.components model"
run(){ local tree=$1 gpu=$2 name=$3 ind=$4 steps=$5; shift 5
  ( source /workspace/venv_bfx9/bin/activate && cd $S/$tree && CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=$ind/triton TORCHINDUCTOR_CACHE_DIR=$ind/inductor TORCHINDUCTOR_COMPILE_THREADS=8 \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$S/$tree timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel $B --training.steps $steps "$@" --dump-folder $S/out_$name > $S/$name.log 2>&1 ); echo "$name rc=$?" >> $S/status2.txt; rm -rf $S/out_$name; }
( run probe 0 probe_warm $S/cache_probe2 1 $C && run probe 0 probe_inductor3 $S/cache_probe2 3 $C ) &
( run probe 1 probe_aot_eager3 $S/cache_aot 3 $C --compile.backend aot_eager ) &
wait; echo ALL-DONE >> $S/status2.txt
