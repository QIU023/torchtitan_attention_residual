#!/bin/bash
# Locate the DEP on/off gap: initial weights with and without DEP, then DEP with the replica's
# init under fork_rng (local probe patch), all on one warm cache.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/dep_fix; P=$R/progress.txt; V=/workspace/venv_bfx9/bin; W=$S/wt_dep_new; D=$S/overnight/dep
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
git -C $W apply $R/shim_5dc97a3e7_pp.patch || { note "t5d shim failed"; exit 1; }
run() {  # <name> <config> <cache> <steps> [extra args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4
  local O=$R/$name; rm -rf $O; mkdir -p $O
  ( cd $W && PYTHONPATH=$R/probe:$D:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 1500 $V/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_p2 --config $cfg --training.steps $steps --metrics.log_freq 1 \
    --dump-folder $O/out --debug.seed 42 --debug.deterministic "$@" > $O/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $O/rc; note "$name rc=$rc"
}
C0=$R/cacheD; rm -rf $C0; mkdir -p $C0
for c in dep_off dep_k25 dep_bubble; do run dwarm_$c $c $C0 1; done
for x in "p_off:dep_off" "p_k25:dep_k25"; do
  n=${x%%:*}; c=${x#*:}; rm -rf $R/cache_c; cp -r $C0 $R/cache_c
  DEP_PARAM_DUMP=$R/$n/params run $n $c $R/cache_c 1
done
git -C $W apply $R/probe/rng_fork_probe.patch || { note "rng probe patch failed"; exit 1; }
for x in "f_off:dep_off" "f_k25:dep_k25" "f_bubble:dep_bubble"; do
  n=${x%%:*}; c=${x#*:}; rm -rf $R/cache_c; cp -r $C0 $R/cache_c
  DEP_PARAM_DUMP=$R/$n/params DEP_GRAD_DUMP=$R/$n/grads DEP_GRAD_STEPS=1,2 run $n $c $R/cache_c 3
done
git -C $W checkout -- torchtitan/models/kimi_k3/pipeline_parallel/__init__.py torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
rm -rf $C0 $R/cache_c
note "T5d done dirty=$(git -C $W status --short | wc -l)"
echo done > $R/T5D_DONE
