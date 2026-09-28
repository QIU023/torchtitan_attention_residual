#!/bin/bash
# G6: DEP step time and a traced step at a size where GPU compute sets the step (dep_scaled.py), 4 x H100.
K=~/k927/kit; R=~/k927/res/dep_scaled; W=~/w/dep; V=~/venv_pp/bin
mkdir -p $R
run() {  # <name> <config> <cache> <steps> [extra...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4
  local D=$R/$name; rm -rf $D; mkdir -p $D/tmp
  ( cd $W && TMPDIR=$D/tmp PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3600 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_scaled --config $cfg --training.steps $steps --metrics.log_freq 1 \
    --training.num-tokens-per-train-step 32768 --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; echo "$(date +%H:%M:%S) $name $(cat $D/rc)" >> $R/progress.txt
}
T0=$R/cache_t0; mkdir -p $T0
for cfg in s_off s_bubble_off s_prefetch s_bubble_on; do run twarm_$cfg $cfg $T0 2; done
for cfg in s_off s_bubble_off s_prefetch s_bubble_on; do
  rm -rf $R/cache_t_$cfg; cp -r $T0 $R/cache_t_$cfg
  run time_$cfg $cfg $R/cache_t_$cfg 30
done
for cfg in s_off s_bubble_off s_bubble_on; do
  rm -rf $R/cache_tr_$cfg; cp -r $T0 $R/cache_tr_$cfg
  run trace_$cfg $cfg $R/cache_tr_$cfg 16 --profiler.enable-profiling --profiler.profile-freq 15 \
    --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder traces
done
rm -rf $R/cache_*
echo done > $R/G6_DONE
