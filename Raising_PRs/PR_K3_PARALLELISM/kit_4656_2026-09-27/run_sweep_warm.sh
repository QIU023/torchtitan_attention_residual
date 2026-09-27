#!/bin/bash
# The AC-off token sweep again, each configuration warmed on its own cache first (reserved memory without autotuning).
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/acr; MAIN=$S/wt_main_f359; PR=$S/wt_attnres
until [ -f $S/lb6/ALL_DONE ]; do sleep 20; done
run() {  # <name> <tree> <gpu> <cache> <steps> <tokens>
  local D=$R/$1; rm -rf $D; mkdir -p $D
  cd $2
  CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$4/ic TRITON_CACHE_DIR=$4/tc \
  /workspace/venv_bfx9/bin/torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
    --debug.seed 42 --debug.deterministic --training.steps $5 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step $6 --training.num-tokens-per-microbatch-per-dp-rank $6 \
    --dump-folder $D/out activation-checkpoint:none > $D/run.log 2>&1
  echo "rc=$?" > $D/rc
}
i=0
for tok in 2048 4096 8192; do for tree in main pr; do
  t=$MAIN; [ $tree = pr ] && t=$PR
  C=$R/cache_sw_${tree}_$tok; rm -rf $C; mkdir -p $C
  ( run swwarm_${tree}_$tok $t $i $C 1 $tok && cp -r $C ${C}_m && run sw_${tree}_$tok $t $i ${C}_m 3 $tok ) &
  i=$((i+1))
done; done
wait
rm -rf $R/cache_sw_*
echo done > $R/SW_DONE
