#!/bin/bash
# Is this PR's +4.5% tps under selective AC real? Warm one cache with a step of each tree, then measure each
# tree twice on its own copy, alternating, one H100, 10 steps, the identity table's setup.
R=~/k928/res/4656_sel; V=~/venv_pp/bin; GPU=${GPU:-0}
declare -A TREE=([main]=$HOME/w/main [pr]=$HOME/w/c4780)
mkdir -p $R
cell() {  # <name> <tree> <cache> <steps>
  local name=$1 tree=$2 cache=$3 steps=$4 D=$R/$1; rm -rf $D; mkdir -p $D/tmp
  ( cd ${TREE[$tree]} && CUDA_VISIBLE_DEVICES=$GPU TMPDIR=$D/tmp TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 1800 $V/torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --training.steps $steps --metrics.log_freq 1 --training.num-tokens-per-train-step 2048 \
    --training.num-tokens-per-microbatch-per-dp-rank 512 --dump-folder $D/out activation-checkpoint:selective > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; echo "$(date +%H:%M:%S) $name $(cat $D/rc)" >> $R/progress.txt; rm -rf $D/tmp $D/out
}
C0=$R/cache0; rm -rf $C0; mkdir -p $C0
cell warm_main main $C0 1; cell warm_pr pr $C0 1
for run in a b; do for t in main pr; do
  rm -rf $R/cache_c; cp -r $C0 $R/cache_c; cell sel_${t}_$run $t $R/cache_c 10
done; done
rm -rf $C0 $R/cache_c
echo done > $R/DONE
