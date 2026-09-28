#!/bin/bash
# #4656 first table at 65536-token micro-batches (four per step), ONE H100, cells one at a time: main f35966713 (~/w/main) against this PR f14d681f4 (~/w/c4780),
# main's kimi_k3_debugmodel recipe as it is: 262144 tokens per step in 65536-token micro-batches, 10 steps,
# none / selective / full; one cache warmed by one step of each configuration, every cell measured on its own
# copy; plus main AC off on a cache of its own (warmed, then measured on a copy).
# Usage: GPU=0 bash run_4656_t1_65536.sh   (results in ~/k928/res/4656_t1)
R=~/k928/res/4656_t1; V=~/venv_pp/bin; GPU=${GPU:-0}
declare -A TREE=([main]=$HOME/w/main [pr]=$HOME/w/c4780)
mkdir -p $R
for t in main pr; do echo "$t $(git -C ${TREE[$t]} rev-parse --short HEAD) dirty=$(git -C ${TREE[$t]} status --short | wc -l)"; done > $R/trees.txt
cell() {  # <name> <tree> <cache> <steps> <tokens per step> <tokens per micro-batch> <ac>
  local name=$1 tree=$2 cache=$3 steps=$4 tps=$5 tpm=$6 ac=$7
  local D=$R/$name; rm -rf $D; mkdir -p $D/tmp
  ( cd ${TREE[$tree]} && CUDA_VISIBLE_DEVICES=$GPU TMPDIR=$D/tmp TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3600 $V/torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --training.steps $steps --metrics.log_freq 1 --training.num-tokens-per-train-step $tps \
    --training.num-tokens-per-microbatch-per-dp-rank $tpm --dump-folder $D/out activation-checkpoint:$ac > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc
  local oom=$(grep -a -ci "out of memory" $D/run.log)
  echo "$(date +%H:%M:%S) $name $(cat $D/rc) oom=$oom" >> $R/progress.txt
  rm -rf $D/tmp $D/out
}
# (a)
C0=$R/cache0; rm -rf $C0; mkdir -p $C0
for t in main pr; do for ac in none selective full; do cell warm_${t}_$ac $t $C0 1 262144 65536 $ac; done; done
for t in main pr; do for ac in none selective full; do
  rm -rf $R/cache_id; cp -r $C0 $R/cache_id
  cell id_${t}_$ac $t $R/cache_id 10 262144 65536 $ac
done; done
F=$R/cache_fresh; rm -rf $F; mkdir -p $F
cell warm_main_none_fresh main $F 1 262144 65536 none
rm -rf $R/cache_id; cp -r $F $R/cache_id
cell id_main_none_fresh main $R/cache_id 10 262144 65536 none
rm -rf $R/cache_id $F
rm -rf $C0
echo done > $R/DONE
