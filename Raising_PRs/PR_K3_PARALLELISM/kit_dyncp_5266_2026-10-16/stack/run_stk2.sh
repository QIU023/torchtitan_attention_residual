#!/bin/bash
# Stack check after folding the fix (8 GPUs): CP2 (dp2 ep2 pp2 cp2) and CP4 (ep4 pp2 cp4), DEP off/on, dynamic CP off/on
# (threshold 256), cc12m-test at native size, the combo seed. A 5-step warm-up of every configuration on one cache, then
# the measured 10-step cells with dumps. No deletion.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; T=$S/wt_mxqat_fix; G=$S/dyncp_fix/stk; O=$G/out
C=$S/k3v_run/combo; SEED=$C/out1/seed/step-0
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
until grep -q "^rc=" $S/dyncp_fix/harness_cp4.log 2>/dev/null; do sleep 20; done
$S/dyncp_fix/run_unit_final.sh
. /workspace/venv_1006/bin/activate
run() {  # <name> <steps> <env...>
  local name=$1 steps=$2; shift 2; local D=$O/$name; mkdir -p $D
  ( cd $T && env "$@" K3C_STEPS=$steps K3C_MAXP=2304 K3C_SEED=$SEED MODULE=stk_probe CONFIG=cell \
    PYTHONPATH=$S/main_pins:$G:$C:$T TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint') dumps=$(ls $D/dump 2>/dev/null | wc -l)"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
[ -d $SEED ] || { note "no seed, stop"; exit 1; }
for phase in warm meas; do
  steps=5; [ $phase = meas ] && steps=10
  for layout in c2 c4; do
    cell="dp=2,ep=2,pp=2,cp=2"; [ $layout = c4 ] && cell="ep=4,pp=2,cp=4"
    for dep in 0 1; do
      for dyn in off on; do
        name=${layout}_dep${dep}_${dyn}_$phase; extra="K3C_CELL=$cell"
        [ $dep = 1 ] && extra="$extra K3C_DEP=1"
        [ $dyn = on ] && extra="$extra K3C_DYNCP=256"
        [ $phase = meas ] && extra="$extra DUMP_DIR=$O/$name/dump"
        run $name $steps $extra
      done
    done
  done
done
note done
