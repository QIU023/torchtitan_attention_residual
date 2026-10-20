#!/bin/bash
# #4380 training table on the H100 (GPUs 0,1): main e61df55b6 twice (all-gather) and once (Ulysses); the PR a2a97b88a with dynamic
# CP off, on with no image reaching the threshold, at 256, at 128, at 128 with main's bank swapped in, and at 256 with Ulysses.
# CP=2, dp1, typechecking on, the recipe's DistMuon, seed 42, deterministic, 100 steps, every cell from one seed checkpoint;
# one cache warmed by 10 steps of every cell.
O=/workspace/h100_train2; mkdir -p $O; K=/workspace/kit_cpmm
echo "$(date +%H:%M:%S) main $(git -C /workspace/tt_main rev-parse --short HEAD) pr $(git -C /workspace/tt_cpmm rev-parse --short HEAD)" >> $O/progress.txt
export KIT_ENV=/workspace/kit_setup/env_cu126.sh PP_PRE=/workspace/kit_setup/shims WARM_STEPS=10 STEPS=100
M=/workspace/tt_main; P=/workspace/tt_cpmm; SEED=$O/seed
if [ ! -d $SEED/step-0 ]; then
  . $KIT_ENV
  ( cd $M && CP_SEED_OUT=$SEED PYTHONPATH=$PP_PRE:$K:. CUDA_VISIBLE_DEVICES=0 NGPU=1 LOG_RANK=0 MODULE=cpmm_probe CONFIG=seed \
      timeout 1800 ./run_train.sh --output-dir $O/seed_run > $O/seed.log 2>&1 )
  echo "$(date +%H:%M:%S) seed rc=$? $(ls $SEED 2>&1 | tr '\n' ' ')" >> $O/progress.txt
fi
[ -d $SEED/step-0 ] || { echo "$(date +%H:%M:%S) no seed, stop" >> $O/progress.txt; exit 1; }
export CP_SEED=$SEED/step-0
$K/run_matrix2.sh $O \
  main_a:$M:0,1:CP_MODE=allgather,WARM=e main_b:$M:0,1:CP_MODE=allgather,WARM=e main_ul:$M:0,1:CP_MODE=ulysses,WARM=e \
  pr_off:$P:0,1:CP_MODE=allgather,CP_MINP=off,WARM=e pr_whole:$P:0,1:CP_MODE=allgather,CP_MINP=1000000000,WARM=e \
  pr_256:$P:0,1:CP_MODE=allgather,CP_MINP=256,WARM=e pr_128:$P:0,1:CP_MODE=allgather,CP_MINP=128,WARM=e \
  pr_128_swap:$P:0,1:CP_MODE=allgather,CP_MINP=128,BANK_SWAP=1,BANK_SWAP_LOG=$O/swap_log,WARM=e \
  pr_256_ul:$P:0,1:CP_MODE=ulysses,CP_MINP=256,WARM=e
echo "$(date +%H:%M:%S) TRAIN_TABLE_DONE" >> $O/progress.txt
