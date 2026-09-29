#!/bin/bash
# Local e2e check of the MoonEP rewrite on 4 x 5060 with the fake MoonEP: the h100 cell
# against the same cell on the standard backend, one warm compile cache, 10 steps each.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/moonep_review; E=$R/e2e; W=$S/wt_moonep_v2; V=/workspace/venv_bfx9/bin; P=$E/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
git -C $W apply $E/shim_5dc97a3e7.patch || { note "shim failed"; exit 1; }
run() {  # <name> <config> <cache> <steps>
  local name=$1 cfg=$2 cache=$3 steps=$4; local D=$E/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$R/fake:$E:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1200 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --role rank --tee 3 -m torchtitan.train --module moonep_local --config $cfg --training.steps $steps \
    --metrics.log_freq 1 --debug.seed 42 --debug.deterministic --dump-folder $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; note "$name rc=$rc"
}
C0=$E/cache0; rm -rf $C0; mkdir -p $C0
run warm standard_cell $C0 1
for x in "std:standard_cell" "moon:moonep_cell"; do
  n=${x%%:*}; c=${x#*:}; rm -rf $E/cache_$n; cp -r $C0 $E/cache_$n; run $n $c $E/cache_$n 10
done
rm -rf $E/cache0 $E/cache_std $E/cache_moon
git -C $W checkout -- torchtitan/distributed/utils.py
note "e2e done dirty_other=$(git -C $W status --short | grep -c utils.py)"
echo done > $E/E2E_DONE
