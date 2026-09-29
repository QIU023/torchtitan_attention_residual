#!/bin/bash
# The 09-29 rebase of the MoonEP rewrite onto main 46ec3f232 (head 30157477b), on 4 x 5060 with the fake MoonEP:
# the h100 cell against the same cell on the standard backend, one warm compile cache, 10 steps each.
# torch 2.15.0.dev20260928 in venv_0928 needs no shim.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=$(cd "$(dirname "$0")" && pwd); F=$(cd "$K/../fake_moonep" && pwd)
E=$S/moonep_rb_e2e; W=$S/wt_moonep_0929n; V=/workspace/venv_0928/bin; P=$E/progress.txt; mkdir -p $E
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "tree $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"
run() {  # <name> <config> <cache> <steps>
  local name=$1 cfg=$2 cache=$3 steps=$4; local D=$E/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$F:$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1200 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --role rank --tee 3 -m torchtitan.train --module moonep_local --config $cfg --training.steps $steps \
    --metrics.log_freq 1 --debug.seed 42 --debug.deterministic --dump-folder $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out; note "$name rc=$rc"
}
C0=$E/cache0; rm -rf $C0; mkdir -p $C0
run warm standard_cell $C0 1
for x in "std:standard_cell" "moon:moonep_cell"; do
  n=${x%%:*}; c=${x#*:}; rm -rf $E/cache_$n; cp -r $C0 $E/cache_$n; run $n $c $E/cache_$n 10; rm -rf $E/cache_$n
done
rm -rf $C0
note "e2e done"
