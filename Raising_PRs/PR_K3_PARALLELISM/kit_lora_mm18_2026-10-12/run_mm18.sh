#!/bin/bash
# The 18-cell multimodal parallelism matrix for one arm on one tree, 8 x 5060, venv_1006 with main's pins first.
# usage: run_mm18.sh <arm: base|lora|qlora> <tree dir> <out dir>
# A seed checkpoint of the arm is written by a single-GPU run, every cell loads it (model only) and must log the load.
# Streams: two 2-GPU streams (0,1 and 6,7; dp1 runs on GPU 0 alone), one 4-GPU stream (2-5), then the 8-GPU cells;
# one cache per stream. ONLY=<cell> reruns one 1-GPU dp1 cell against an existing seed (no new seed).
ARM=$1 T=$2 O=$3
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <cache> <env...>
  local name=$1 gpus=$2 cache=$3; shift 3; local D=$O/$name; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" MM18_ARM=$ARM PYTHONPATH=$S/main_pins:$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint')"
}
note "arm $ARM tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l) torch $(python -c 'import torch;print(torch.__version__)')"
SEED=$O/seed
if [ "${ONLY:-}" = dp1 ]; then
  run dp1 0 $O/cache_a MM18_CELL=dp=1 MM18_SEED=$SEED/step-0 MODULE=mm18_probe CONFIG=cell
  exit 0
fi
rm -rf $SEED
run seed 0 $O/cache_seed MM18_SEED_OUT=$SEED MODULE=mm18_probe CONFIG=seed MM18_STEPS=1
ls -d $SEED/step-0 >/dev/null 2>&1 || { note "no seed at $SEED/step-0, stop"; echo failed > $O/DONE; exit 1; }
cellrun() { run "$1" "$2" "$3" MM18_CELL="$4" MM18_SEED=$SEED/step-0 MODULE=mm18_probe CONFIG=cell; }
streamA() { cellrun dp1 0 $O/cache_a "dp=1"; for c in "fsdp2|dp=2" "pp2|pp=2"; do cellrun ${c%%|*} 0,1 $O/cache_a "${c#*|}"; done; }
streamA2() { for c in "tp2_ep2|tp=2,ep=2" "ep2_fsdp2|dp=2,ep=2" "cp2|cp=2"; do cellrun ${c%%|*} 6,7 $O/cache_a2 "${c#*|}"; done; }
streamB() { for c in "pp4|pp=4" "tp4_ep4|tp=4,ep=4" "cp4|cp=4"; do cellrun ${c%%|*} 2,3,4,5 $O/cache_b "${c#*|}"; done; }
streamA & streamA2 & streamB & wait
for c in "pp8|pp=8" "ep8_fsdp8|dp=8,ep=8" "ep4_fsdp2_tp2_pp2|dp=2,tp=2,ep=4,pp=2" "ep2_fsdp2_tp2_pp2|dp=2,tp=2,ep=2,pp=2" \
         "fsdp2_pp2_cp2|dp=2,pp=2,cp=2" "ep2_fsdp2_pp2_cp2|dp=2,ep=2,pp=2,cp=2" "ep4_fsdp2_tp2_cp2|dp=2,tp=2,ep=4,cp=2" \
         "ep2_tp2_pp2_cp2|tp=2,ep=2,pp=2,cp=2" "ep2_fsdp2_tp2_cp2|dp=2,tp=2,ep=2,cp=2"; do
  cellrun ${c%%|*} 0,1,2,3,4,5,6,7 $O/cache_c "${c#*|}"
done
rm -rf $O/cache_*
note "done"; echo done > $O/DONE
