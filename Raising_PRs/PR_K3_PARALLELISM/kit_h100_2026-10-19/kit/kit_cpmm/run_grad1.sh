#!/bin/bash
# Step-1 gradients of the #4380 table cells (GPUs 0,1), each on a copy of the table's warm cache cache_e, from the same seed;
# the step-1 loss must match the table's before a dump counts.
O=/workspace/h100_grad1; mkdir -p $O; K=/workspace/kit_cpmm; T=/workspace/h100_train2; M=/workspace/tt_main; P=/workspace/tt_cpmm
. /workspace/kit_setup/env_cu126.sh
note() { echo "$(date +%H:%M:%S) $*" >> $O/progress.txt; }
for spec in main_a:$M:CP_MODE=allgather pr_off:$P:CP_MODE=allgather,CP_MINP=off \
            pr_whole:$P:CP_MODE=allgather,CP_MINP=1000000000 pr_128:$P:CP_MODE=allgather,CP_MINP=128 \
            pr_128_swap:$P:CP_MODE=allgather,CP_MINP=128,BANK_SWAP=1,BANK_SWAP_LOG=$O/swap_log; do
  IFS=: read name W envs <<< "$spec"; D=$O/$name; rm -rf $D; mkdir -p $D; rm -rf $O/cc; cp -r $T/cache_e $O/cc
  ( cd $W && env $(echo $envs | tr ',' ' ') CP_STEPS=1 CP_SEED=$T/seed/step-0 DUMP_DIR=$D/dump \
      PYTHONPATH=/workspace/kit_setup/shims:$K:. TORCHINDUCTOR_CACHE_DIR=$O/cc/ic TRITON_CACHE_DIR=$O/cc/tc \
      CUDA_VISIBLE_DEVICES=0,1 NGPU=2 LOG_RANK=0 MODULE=grad1_probe CONFIG=cell \
      timeout 1800 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  rc=$?; rm -rf $D/out
  note "$name rc=$rc patched=$(grep -a -c 'optim_step patched' $D/run.log) dumps=$(ls $D/dump 2>/dev/null | wc -l) $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *1 *loss: *[0-9.]* *grad_norm: *[0-9.]*' | head -1)"
done
rm -rf $O/cc
note GRAD1_DONE
