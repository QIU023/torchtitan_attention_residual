#!/bin/bash
# After matrix_1017 finishes: rerun cp2 (stream A2 cache) and cp4 (stream B cache), which failed on the kit's old CP class name.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; R=$S/k3v_run; O=$R/matrix_1017; T=$S/wt_k3_video
until grep -q " done$" $O/progress.txt; do sleep 20; done
. /workspace/venv_1006/bin/activate
for spec in "cp2|cp=2|6,7|A2" "cp4|cp=4|2,3,4,5|B"; do
  IFS='|' read -r name deg gpus s <<< "$spec"; mv $O/$name $O/${name}_void_classname; D=$O/$name; mkdir -p $D
  n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && K3V_CELL=$deg K3V_SEED=$O/seed/step-0 MODULE=k3v_cells CONFIG=cell PYTHONPATH=$S/main_pins_1016:$R:$T TORCHINDUCTOR_CACHE_DIR=$O/cache_$s/ic TRITON_CACHE_DIR=$O/cache_$s/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) MASTER_PORT=$((29500 + ${gpus%%,*})) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  rc=$?; clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  echo "$(date +%H:%M:%S) rerun $name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint')" >> $O/progress.txt
done
echo "$(date +%H:%M:%S) rerun done" >> $O/progress.txt
