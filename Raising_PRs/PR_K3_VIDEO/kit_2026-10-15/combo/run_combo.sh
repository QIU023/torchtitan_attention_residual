#!/bin/bash
# DEP and dynamic CP together on the stack: seed, then four 8-GPU cells in sequence on one cache. No deletion.
T=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_mxqat_fix
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; R=$S/k3v_run/combo; O=$R/out1
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <env...>
  local name=$1 gpus=$2; shift 2; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" PYTHONPATH=$S/main_pins:$R:$T TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint')"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
SEED=$O/seed
[ -d $SEED/step-0 ] || run seed 0 K3C_SEED_OUT=$SEED K3C_MAXP=2304 MODULE=combo_cells CONFIG=seed
[ -d $SEED/step-0 ] || { note "no seed, stop"; exit 1; }
G=0,1,2,3,4,5,6,7; C="dp=2,ep=2,pp=2,cp=2"
run control $G K3C_CELL=$C K3C_MAXP=2304 K3C_SEED=$SEED/step-0 MODULE=combo_cells CONFIG=cell
run dep $G K3C_CELL=$C K3C_MAXP=2304 K3C_DEP=1 K3C_SEED=$SEED/step-0 MODULE=combo_cells CONFIG=cell
run dyncp $G K3C_CELL=$C K3C_MAXP=2304 K3C_DYNCP=256 K3C_SEED=$SEED/step-0 MODULE=combo_cells CONFIG=cell
run dep_dyncp $G K3C_CELL=$C K3C_MAXP=2304 K3C_DEP=1 K3C_DYNCP=256 K3C_SEED=$SEED/step-0 MODULE=combo_cells CONFIG=cell
note done
