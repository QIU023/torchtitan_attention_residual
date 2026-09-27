#!/bin/bash
# #4656 on 5060: identity and memory against main, one cache lineage, then the PP composition smoke.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/acr; MAIN=$S/wt_main_f359; PR=$S/wt_attnres
C0=$R/cache0; mkdir -p $C0
run() {  # <name> <tree> <gpu> <cache> <steps> <tokens/step> <tokens/mb> <ac>
  local D=$R/$1; rm -rf $D; mkdir -p $D
  cd $2
  CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$4/ic TRITON_CACHE_DIR=$4/tc \
  /workspace/venv_bfx9/bin/torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
    --debug.seed 42 --debug.deterministic --training.steps $5 --metrics.log_freq 1 \
    --training.num-tokens-per-train-step $6 --training.num-tokens-per-microbatch-per-dp-rank $7 \
    --dump-folder $D/out activation-checkpoint:$8 > $D/run.log 2>&1
  echo "rc=$?" > $D/rc
}
# warm cache0 with one step of every configuration
for ac in none selective full; do
  run warm_main_$ac $MAIN 0 $C0 1 2048 512 $ac
  run warm_pr_$ac $PR 0 $C0 1 2048 512 $ac
done
# measured identity cells, each on its own copy of cache0, in parallel
i=0
for tree in main pr; do for ac in none selective full; do
  cp -r $C0 $R/cache_${tree}_$ac
  t=$MAIN; [ $tree = pr ] && t=$PR
  run id_${tree}_$ac $t $i $R/cache_${tree}_$ac 10 2048 512 $ac &
  i=$((i+1))
done; done
mkdir -p $R/cache_fresh
run id_main_none_fresh $MAIN 6 $R/cache_fresh 10 2048 512 none &
wait
echo done > $R/ID_DONE
# memory scaling, AC off, one micro-batch per step, 3 steps, fresh caches (memory only)
i=0
for tok in 2048 4096 8192; do for tree in main pr; do
  t=$MAIN; [ $tree = pr ] && t=$PR
  mkdir -p $R/cache_sc_${tree}_$tok
  run sc_${tree}_$tok $t $i $R/cache_sc_${tree}_$tok 3 $tok $tok none &
  i=$((i+1))
done; done
wait
echo done > $R/SC_DONE
rm -rf $R/cache_*
