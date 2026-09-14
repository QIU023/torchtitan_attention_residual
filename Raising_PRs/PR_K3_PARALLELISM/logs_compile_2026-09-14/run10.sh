#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/compile_fix
until grep -q ALL-DONE $S/status9.txt 2>/dev/null; do sleep 15; done
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
( source /workspace/venv_bfx9/bin/activate && cd $S/probe4 && CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$S/cache_33/triton TORCHINDUCTOR_CACHE_DIR=$S/cache_33/inductor TORCHINDUCTOR_COMPILE_THREADS=8 TORCH_LOGS=recompiles \
  PYTHONPATH=/workspace/pylib/attn_gym_main:$S/probe4:$S/k3scratch_pkg timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
  -m torchtitan.train --module k3scratch --config kimi_k3_debugmodel33 $B --training.steps 3 --compile.enable --compile.components model --dump-folder $S/out_c33 > $S/h_compiled33.log 2>&1 ); echo "h_compiled33 rc=$?" >> $S/status10.txt; rm -rf $S/out_c33
