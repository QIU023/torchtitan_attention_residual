#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/compile_fix
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
( source /workspace/venv_bfx9/bin/activate && cd $S/probe5 && CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=$S/cache_w/triton TORCHINDUCTOR_CACHE_DIR=$S/cache_w/inductor TORCHINDUCTOR_COMPILE_THREADS=8 \
  PROBE_LAYER=23 GRAD_FILTER=layers.23. PROBE_TXT=1 PROBE_REF=$S/tapref.pt GRAD_DUMP_TXT=$S/t_aot.txt GRAD_REF=$S/gref23 \
  PYTHONPATH=/workspace/pylib/attn_gym_main:$S/probe5 timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
  -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel $B --training.steps 2 --compile.enable --compile.components model --compile.backend aot_eager --dump-folder $S/out_t_aot2 > $S/t_aot_eager2.log 2>&1 ); echo "t_aot_eager2 rc=$?" >> $S/status7.txt; rm -rf $S/out_t_aot2
