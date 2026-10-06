#!/bin/bash
# Usage: run_cells.sh <worktree> <out_dir> <name>:<gpus>:<env...> ...  (one cell at a time, one shared cache per tree)
W=$1; O=$2; shift 2; K=$(cd "$(dirname "$0")" && pwd); mkdir -p $O; C=$O/cache; mkdir -p $C/ic $C/tc
. /workspace/venv_1003b/bin/activate
for spec in "$@"; do
  name=${spec%%:*}; rest=${spec#*:}; gpus=${rest%%:*}; envs=${rest#*:}; n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $W && env $envs PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc CUDA_VISIBLE_DEVICES=$gpus \
    NGPU=$n LOG_RANK=0 MODULE=cpmm_probe CONFIG=cell timeout 2400 ./run_train.sh --output-dir $O/out_$name > $O/$name.log 2>&1 )
  rc=$?; echo "$(date +%H:%M:%S) $name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $O/$name.log | grep -a -o 'step: *[0-9]* *loss: *[0-9.]*' | tail -1)" >> $O/progress.txt
done
echo done >> $O/progress.txt
