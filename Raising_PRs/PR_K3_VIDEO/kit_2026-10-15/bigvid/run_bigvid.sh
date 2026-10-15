#!/bin/bash
# Video at 1792 px a frame on the stack plus the video commit (scratch worktree): seed, then pp2 x cp4 x ep4 cells
# (control, DEP, dynamic CP, both) in sequence, one cache, flex compiled without autotune (5060 shim). No deletion.
T=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_stack_video_try
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; BV=$S/k3v_run/bigvid; O=$BV/${OUT:-out1}
NOAT=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_cpmm_2026-10-06/local/flex_no_autotune
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <env...>
  local name=$1 gpus=$2; shift 2; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" PYTHONPATH=$NOAT:$S/main_pins:$BV:$T TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 1500 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint') noat=$(grep -c 'flex_no_autotune' $D/run.log)"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
SEED=$O/seed
[ -d $SEED/step-0 ] || run seed 0 K3V_SEED_OUT=$SEED MODULE=bigvid_cells CONFIG=seed
[ -d $SEED/step-0 ] || { note "no seed, stop"; exit 1; }
G=0,1,2,3,4,5,6,7; A="pp=2,cp=4,ep=4"
run A_control $G K3V_CELL=$A K3V_SEED=$SEED/step-0 MODULE=bigvid_cells CONFIG=cell
run A_dep $G K3V_CELL=$A K3V_DEP=1 K3V_SEED=$SEED/step-0 MODULE=bigvid_cells CONFIG=cell
run A_dyncp $G K3V_CELL=$A K3V_DYNCP=256 K3V_SEED=$SEED/step-0 MODULE=bigvid_cells CONFIG=cell
run A_dep_dyncp $G K3V_CELL=$A K3V_DEP=1 K3V_DYNCP=256 K3V_SEED=$SEED/step-0 MODULE=bigvid_cells CONFIG=cell
note done
