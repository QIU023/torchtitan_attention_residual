#!/bin/bash
# Seeded B200 cell: seed on one GPU, then main DEP off, branch DEP off, branch DEP on, 8 GPUs each, one cache.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; D=$S/dep_rebase_1019; W=$S/wt_dep_tests; M=$S/wt_main_check
O=$D/seeded; mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <tree> <gpus> <env...>
  local name=$1 tree=$2 gpus=$3; shift 3; local R=$O/$name; mkdir -p $R
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $tree && env "$@" K3_FORCE_FLEX=1 MODULE=dep_cells PYTHONPATH=$D/flexforce:$D:$S/main_pins_1016:$tree TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $R/out > $R/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $R/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint') dep=$(echo "$clean" | grep -c 'vision encodes decoupled') flex=$(echo "$clean" | grep -c 'K3_FORCE_FLEX active')"
}
note "branch $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l); main $(git -C $M rev-parse --short HEAD) dirty=$(git -C $M status --short | wc -l)"
[ -d $O/seed/step-0 ] || run seed $W 0 CONFIG=seed DEP_SEED_OUT=$O/seed
[ -d $O/seed/step-0 ] || { note "no seed, stop"; exit 1; }
G=0,1,2,3,4,5,6,7
run main_off $M $G CONFIG=cell DEP_SEED=$O/seed/step-0 DEP_OFF=1
run branch_off $W $G CONFIG=cell DEP_SEED=$O/seed/step-0 DEP_OFF=1
run branch_on $W $G CONFIG=cell DEP_SEED=$O/seed/step-0
note done
