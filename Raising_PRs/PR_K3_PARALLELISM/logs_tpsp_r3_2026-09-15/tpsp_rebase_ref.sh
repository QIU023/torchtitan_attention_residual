#!/bin/bash
# Did the rebase move the tp=1 reference? old base 56a721b64 vs new main d34a13fdf vs PR head, 1 GPU, body flags.
OLD=/tmp/wt_main_56a721b64 NEW=/tmp/wt_main_d34a13fdf PR=/tmp/wt_tpsp_review4
OUT=/workspace/.tpsp_r3/ref_$(date +%m%d_%H%M%S); mkdir -p $OUT/cache_old $OUT/cache_shared; R=$OUT/results.txt
echo "old=$(git -C $OLD rev-parse --short HEAD) new=$(git -C $NEW rev-parse --short HEAD) pr=$(git -C $PR rev-parse --short HEAD)" > $R
restore(){ for t in $OLD $NEW $PR; do git -C $t checkout -q -- torchtitan/models/kimi_k3/kda.py; done; }
trap restore EXIT
for t in $OLD $NEW $PR; do sed -i 's/if capability not in {(10, 0), (10, 3)}:/if capability < (8, 0):  # LOCAL GUARD LIFT/' $t/torchtitan/models/kimi_k3/kda.py
  grep -q "LOCAL GUARD LIFT" $t/torchtitan/models/kimi_k3/kda.py || { echo "guard lift failed $t" >> $R; exit 1; }; done
run(){ local nm=$1 tree=$2 cache=$3 steps=$4
  ( source /workspace/venv_bfx9/bin/activate && cd $tree && CUDA_VISIBLE_DEVICES=0 NGPU=1 LOG_RANK=0 MODULE=kimi_k3 CONFIG=kimi_k3_debugmodel \
    TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton PYTHONPATH=/workspace/pylib/attn_gym_main:$tree timeout 2400 ./run_train.sh \
    --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps $steps \
    --training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256 ) > $OUT/$nm.log 2>&1
  echo "$nm rc=$?" >> $R; sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -E "step: *[0-9]+" | grep -o "step:.*grad_norm: *[0-9.]*" | sed 's/^/  /' >> $R
  sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -m2 -E "Traceback|Error:" | sed 's/^/  ! /' >> $R; }
run warm_new $NEW $OUT/cache_shared 1; run warm_pr $PR $OUT/cache_shared 1; run warm_old $OLD $OUT/cache_old 1
run old_base $OLD $OUT/cache_old 3; run new_main $NEW $OUT/cache_shared 3; run pr_head $PR $OUT/cache_shared 3
echo "DONE $OUT" >> $R; echo "DONE $OUT"
