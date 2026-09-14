#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/compile_fix
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
C="--compile.enable --compile.components model"
run(){ local gpu=$1 name=$2 cache=$3 steps=$4; shift 4
  ( source /workspace/venv_bfx9/bin/activate && cd $S/probe4 && CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=$cache/triton TORCHINDUCTOR_CACHE_DIR=$cache/inductor TORCHINDUCTOR_COMPILE_THREADS=8 \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$S/probe4 timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel $B --training.steps $steps "$@" --dump-folder $S/out_$name > $S/$name.log 2>&1 ); echo "$name rc=$?" >> $S/status4.txt; rm -rf $S/out_$name; }
(
  GRAD_DUMP_TXT=$S/g_eager.txt GRAD_SAVE=$S/gref run 1 h_eager $S/cache_w 3
  GRAD_DUMP_TXT=$S/g_aot.txt GRAD_REF=$S/gref run 1 h_aot_eager $S/cache_w 3 $C --compile.backend aot_eager
  GRAD_DUMP_TXT=$S/g_noise2.txt GRAD_REF=$S/gref run 1 h_eager_fresh2 $S/cache_n2 3
) &
(
  run 0 h_warm $S/cache_w 1 $C
  until [ -f $S/gref/.done ] || grep -q "h_eager rc=[1-9]" $S/status4.txt 2>/dev/null; do sleep 10; done
  TORCH_LOGS=dynamo GRAD_DUMP_TXT=$S/g_inductor.txt GRAD_REF=$S/gref run 0 h_compiled $S/cache_w 3 $C
  GRAD_DUMP_TXT=$S/g_noise1.txt GRAD_REF=$S/gref run 0 h_eager_fresh1 $S/cache_n1 3
) &
wait; echo ALL-DONE >> $S/status4.txt
