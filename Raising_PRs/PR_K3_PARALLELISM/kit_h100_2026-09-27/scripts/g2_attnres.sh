#!/bin/bash
# G2: #4656 (list, ~/w/l4656) and #4780 (checkpoint, ~/w/c4780) against main (~/w/main), one H100 per cell.
K=~/k927/kit; R=~/k927/res/attnres; V=~/venv_pp/bin
declare -A TREE=([main]=~/w/main [l4656]=~/w/l4656 [c4780]=~/w/c4780)
mkdir -p $R
cell() {  # <name> <tree> <gpu> <cache> <module> <config> <steps> <tokens/step> <tokens/mb> <ac> [VAR=val ...]
  local name=$1 tree=$2 gpu=$3 cache=$4 module=$5 cfg=$6 steps=$7 tps=$8 tpm=$9 ac=${10}; shift 10
  local D=$R/$name; rm -rf $D; mkdir -p $D/tmp
  ( cd ${TREE[$tree]} && env "$@" CUDA_VISIBLE_DEVICES=$gpu TMPDIR=$D/tmp PYTHONPATH=$K:. \
    TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3600 $V/torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module $module --config $cfg --debug.seed 42 --debug.deterministic \
    --training.steps $steps --metrics.log_freq 1 --training.num-tokens-per-train-step $tps \
    --training.num-tokens-per-microbatch-per-dp-rank $tpm --dump-folder $D/out activation-checkpoint:$ac > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc
  echo "$(date +%H:%M:%S) $name $(cat $D/rc)" >> $R/progress.txt
}
# (a) identity on the body's recipe: one shared cache, warmed by one step of each configuration
C0=$R/cache0; mkdir -p $C0
for t in main l4656 c4780; do for ac in none selective full; do
  cell warm_${t}_$ac $t 0 $C0 kimi_k3 kimi_k3_debugmodel 1 2048 512 $ac
done; done
specs=(); for t in main l4656 c4780; do for ac in none selective full; do specs+=("$t:$ac"); done; done
i=0
for s in "${specs[@]}"; do
  t=${s%%:*}; ac=${s#*:}; rm -rf $R/cache_id_${t}_$ac; cp -r $C0 $R/cache_id_${t}_$ac
  cell id_${t}_$ac $t $((i % 4)) $R/cache_id_${t}_$ac kimi_k3 kimi_k3_debugmodel 10 2048 512 $ac &
  i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done
mkdir -p $R/cache_fresh
cell id_main_none_fresh main $((i % 4)) $R/cache_fresh kimi_k3 kimi_k3_debugmodel 10 2048 512 none &
wait
# (b)-(d): each configuration warms its own cache with one step, then measures three steps on a copy; four at a time
wave() {  # "<name>|<tree>|<module>|<config>|<tokens>|<ac>|<VAR=val,VAR=val>" x up to 4
  local i=0
  for spec in "$@"; do
    IFS='|' read -r name t module cfg tok ac envs <<< "$spec"
    ( W=$R/cache_w_$name; rm -rf $W; mkdir -p $W
      cell w_$name $t $i $W $module $cfg 1 $tok $tok $ac ${envs//,/ }
      rm -rf ${W}_m; cp -r $W ${W}_m
      cell $name $t $i ${W}_m $module $cfg 3 $tok $tok $ac ${envs//,/ }
      rm -rf $W ${W}_m ) &
    i=$((i + 1))
  done
  wait
}
# (b) the debug model, AC off, one micro-batch; (c) the list: main against #4656 under selective and
# full AC, 48 layers at dim 2048, blocks of 4 and 12; (d) the checkpoint: #4656 against #4780 with
# AC off, 24 and 48 layers at dim 2048, blocks of 12. Run four specs at a time.
all=()
for tok in 2048 4096 8192 16384; do for t in main l4656 c4780; do
  all+=("sw_${t}_$tok|$t|kimi_k3|kimi_k3_debugmodel|$tok|none|X=0")
done; done
for ac in selective full; do for blk in 4 12; do for tok in 8192 16384; do for t in main l4656; do
  all+=("ls_${t}_${ac}_b${blk}_$tok|$t|attnres_h100|attnres_scaled|$tok|$ac|ATTNRES_DIM=2048,ATTNRES_LAYERS=48,ATTNRES_BLOCK=$blk")
done; done; done; done
for L in 24 48; do for tok in 4096 8192 16384; do for t in l4656 c4780; do
  all+=("ck_${t}_l${L}_$tok|$t|attnres_h100|attnres_scaled|$tok|none|ATTNRES_DIM=2048,ATTNRES_LAYERS=$L,ATTNRES_BLOCK=12")
done; done; done
for ((j = 0; j < ${#all[@]}; j += 4)); do wave "${all[@]:j:4}"; done
rm -rf $R/cache_*
echo done > $R/G2_DONE
