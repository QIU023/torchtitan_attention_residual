#!/bin/bash
# MoonEP load-balance cells: the Kimi K3 debug model with LOAD_E experts (default 128), top-LOAD_K (default 8), FSDP4 x EP4,
# seq 512, standard against MoonEP, natural routing and a Zipf-like skew (LOAD_SKEW, with the expert-bias update off).
# Numerics and load: seed 42, deterministic, 20 steps, one warm cache. Timing (MODE=real only): 30 steps, not deterministic.
# MODE=fake: 4 x 5060 with the fake MoonEP package (bookkeeping only; its routing and timing mean nothing).
# MODE=real: 4 x H100 behind an NVSwitch with MoonEP 33327eb.
# Env: MODE, TREE (a worktree at moonep_review1 ab191a771), VENV, OUT, GPUS (default 0,1,2,3), SKEW (default 0.05).
set -u
K=$(cd "$(dirname "$0")" && pwd)
MODE=${MODE:-fake}; SKEW=${SKEW:-0.05}; GPUS=${GPUS:-0,1,2,3}
: "${TREE:?}" "${VENV:?}" "${OUT:?}"
mkdir -p $OUT; P=$OUT/progress.txt
EXTRA=""
[ "$MODE" = fake ] && EXTRA=$K/../../kit_moonep_rewrite_2026-09-29/local/fake_moonep
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "mode $MODE tree $(git -C $TREE rev-parse --short HEAD) torch $($VENV/bin/python -c 'import torch; print(torch.__version__)')"
run() {  # <name> <config> <cache> <steps> <skew> [args...]
  local name=$1 cfg=$2 cache=$3 steps=$4 skew=$5; shift 5; local D=$OUT/$name; rm -rf $D; mkdir -p $D
  local nobias=0; [ "$skew" != 0 ] && nobias=1
  ( cd $TREE && CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$K:$EXTRA:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    LOAD_OUT=$D/load LOAD_SKEW=$skew LOAD_NO_BIAS=$nobias \
    timeout 1800 $VENV/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module moonep_load --config $cfg --training.steps $steps --metrics.log-freq 1 \
    --debug.seed 42 --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out; note "$name rc=$rc"
}
C0=$OUT/cache0; rm -rf $C0; mkdir -p $C0
for x in std:load_std moonep:load_moonep; do run warm_${x%%:*} ${x#*:} $C0 1 0 --debug.deterministic; done
for skew in 0 $SKEW; do
  tag=nat; [ "$skew" != 0 ] && tag=skew
  for x in std:load_std moonep:load_moonep; do
    rm -rf $OUT/cache_c; cp -r $C0 $OUT/cache_c
    run num_${x%%:*}_$tag ${x#*:} $OUT/cache_c 20 $skew --debug.deterministic
  done
done
if [ "$MODE" = real ]; then
  for skew in 0 $SKEW; do
    tag=nat; [ "$skew" != 0 ] && tag=skew
    for x in std:load_std moonep:load_moonep; do
      rm -rf $OUT/cache_c; cp -r $C0 $OUT/cache_c
      run time_${x%%:*}_$tag ${x#*:} $OUT/cache_c 30 $skew
    done
  done
fi
rm -rf $OUT/cache_c $C0
$VENV/bin/python $K/tab_load.py $OUT > $OUT/tables.md 2>&1
note "load cells done"; echo done > $OUT/DONE
