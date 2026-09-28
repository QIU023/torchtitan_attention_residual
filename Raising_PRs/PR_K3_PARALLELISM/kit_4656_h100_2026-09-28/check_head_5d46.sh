#!/bin/bash
# The minimal #4656 head 5d469fdf3 against the measured head f14d681f4, one H100, the first table's setup
# (262144 tokens per step in 65536-token micro-batches, 10 steps), AC none and selective, one warm cache.
R=~/k928/res/4656_5d46; V=~/venv_pp/bin; GPU=${GPU:-0}
git -C ~/tt fetch -q origin attnres_review1
[ -d ~/w/v2_4656 ] || git -C ~/tt worktree add -q --detach ~/w/v2_4656 5d469fdf3
declare -A TREE=([old]=$HOME/w/c4780 [new]=$HOME/w/v2_4656)
mkdir -p $R
for t in old new; do echo "$t $(git -C ${TREE[$t]} rev-parse --short HEAD) dirty=$(git -C ${TREE[$t]} status --short | wc -l)"; done > $R/trees.txt
cell() {  # <name> <tree> <cache> <steps> <ac>
  local name=$1 tree=$2 cache=$3 steps=$4 ac=$5 D=$R/$1; rm -rf $D; mkdir -p $D/tmp
  ( cd ${TREE[$tree]} && CUDA_VISIBLE_DEVICES=$GPU TMPDIR=$D/tmp TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 1800 $V/torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --training.steps $steps --metrics.log_freq 1 --training.num-tokens-per-train-step 262144 \
    --training.num-tokens-per-microbatch-per-dp-rank 65536 --dump-folder $D/out activation-checkpoint:$ac > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; echo "$(date +%H:%M:%S) $name $(cat $D/rc)" >> $R/progress.txt; rm -rf $D/tmp $D/out
}
C0=$R/cache0; rm -rf $C0; mkdir -p $C0
for t in old new; do for ac in none selective; do cell warm_${t}_$ac $t $C0 1 $ac; done; done
for t in old new; do for ac in none selective; do rm -rf $R/cache_c; cp -r $C0 $R/cache_c; cell id_${t}_$ac $t $R/cache_c 10 $ac; done; done
rm -rf $C0 $R/cache_c
echo done > $R/DONE
