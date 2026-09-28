#!/bin/bash
# G1: DEP (tree ~/w/dep = 31f372593) on 4 x H100, torch 2.15.0.dev20260926+cu130, no compat shim.
K=~/k927/kit; R=~/k927/res/dep; W=~/w/dep; V=~/venv_pp/bin
mkdir -p $R
run() {  # <name> <module> <config> <cache> <steps> [extra args...]
  local name=$1 module=$2 cfg=$3 cache=$4 steps=$5; shift 5
  local D=$R/$name; rm -rf $D; mkdir -p $D/tmp
  ( cd $W && DEP_GRAD_DUMP=${DEP_GRAD_DUMP:-} TMPDIR=$D/tmp PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3600 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module $module --config $cfg --training.steps $steps --metrics.log_freq 1 \
    --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc
  echo "$(date +%H:%M:%S) $name $(cat $D/rc)" >> $R/progress.txt
}
SEED="--debug.seed 42 --debug.deterministic"
# 1. the committed B200 cell, unseeded
run cell torchtitan_recipes.tests.b200 kimi_k3_debugmodel_pp4_vp2_vit_dep $R/cache_cell 10
# 2. bubble on / off numerics: one warm cache, each setting twice, then step-1..3 gradients
C0=$R/cache0; mkdir -p $C0
run warm_on dep_bubble dep_bubble_on $C0 1 $SEED
run warm_off dep_bubble dep_bubble_off $C0 1 $SEED
for c in on off on2 off2 probe_on probe_off h100_on h100_off; do rm -rf $R/cache_$c; cp -r $C0 $R/cache_$c; done
run on dep_bubble dep_bubble_on $R/cache_on 10 $SEED
run off dep_bubble dep_bubble_off $R/cache_off 10 $SEED
run on2 dep_bubble dep_bubble_on $R/cache_on2 10 $SEED
run off2 dep_bubble dep_bubble_off $R/cache_off2 10 $SEED
mkdir -p $R/probe_on_grads $R/probe_off_grads
DEP_GRAD_DUMP=$R/probe_on_grads run probe_on dep_bubble_probe probe_on $R/cache_probe_on 3 $SEED
DEP_GRAD_DUMP=$R/probe_off_grads run probe_off dep_bubble_probe probe_off $R/cache_probe_off 3 $SEED
$V/python $K/cmp_grads.py $R/probe_on_grads $R/probe_off_grads > $R/cmp_grads.txt 2>&1
# 3. hundred steps, bubble on / off, same lineage
run h100_on dep_bubble dep_bubble_on $R/cache_h100_on 100 $SEED
run h100_off dep_bubble dep_bubble_off $R/cache_h100_off 100 $SEED
# 4. dp_shard 2 x pp2, even DP ranks text only, default reshard and always
run mixed dep_bubble dep_mixed_pp2 $R/cache_mixed 4
run mixed_always dep_bubble dep_mixed_pp2 $R/cache_mixed_always 4 --parallelism.fsdp-reshard-after-forward always
# 5. step time: DEP off / on inline / prefetch / bubble on one warm cache, then a traced step of each
T0=$R/cache_t0; mkdir -p $T0
for cfg in dep_off dep_bubble_off dep_prefetch dep_bubble_on; do run twarm_$cfg dep_bubble $cfg $T0 2; done
for cfg in dep_off dep_bubble_off dep_prefetch dep_bubble_on; do
  rm -rf $R/cache_t_$cfg; cp -r $T0 $R/cache_t_$cfg
  run time_$cfg dep_bubble $cfg $R/cache_t_$cfg 30
done
for cfg in dep_off dep_bubble_off dep_bubble_on; do
  rm -rf $R/cache_tr_$cfg; cp -r $T0 $R/cache_tr_$cfg
  run trace_$cfg dep_bubble $cfg $R/cache_tr_$cfg 16 --profiler.enable-profiling --profiler.profile-freq 15 \
    --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder traces
done
rm -rf $R/cache_*
echo done > $R/G1_DONE
