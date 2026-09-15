#!/bin/bash
# PR 4499 round 3: routed_down under EP, before (ee5b16b0a) vs after (1dec3ee17), fsdp2 x tp2 x ep2 on 4 GPUs.
BEFORE=/tmp/wt_tpsp_before AFTER=/tmp/wt_tpsp_review4
OUT=/workspace/.tpsp_r3/$(date +%m%d_%H%M%S); mkdir -p $OUT/cache; R=$OUT/results.txt
echo "before=$(git -C $BEFORE rev-parse --short HEAD) after=$(git -C $AFTER rev-parse --short HEAD)" > $R
restore(){ for t in $BEFORE $AFTER; do git -C $t checkout -q -- torchtitan/models/kimi_k3/kda.py; done; }
trap restore EXIT
for t in $BEFORE $AFTER; do sed -i 's/if capability not in {(10, 0), (10, 3)}:/if capability < (8, 0):  # LOCAL GUARD LIFT/' $t/torchtitan/models/kimi_k3/kda.py
  grep -q "LOCAL GUARD LIFT" $t/torchtitan/models/kimi_k3/kda.py || { echo "guard lift failed $t" >> $R; exit 1; }; done
run(){ local nm=$1 tree=$2 steps=$3; shift 3
  ( source /workspace/venv_bfx9/bin/activate && cd $tree && CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 LOG_RANK=0 \
    MODULE=torchtitan_recipes.tests.b200 CONFIG=kimi_k3_debugmodel_mm \
    TORCHINDUCTOR_CACHE_DIR=$OUT/cache/inductor TRITON_CACHE_DIR=$OUT/cache/triton \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$tree timeout 2400 ./run_train.sh \
    --training.steps $steps --debug.seed 42 --debug.deterministic --metrics.log_freq 1 "$@" ) > $OUT/$nm.log 2>&1
  echo "$nm rc=$?" >> $R; grep -o "step: *[0-9]* *loss: *[0-9.]* *grad_norm: *[0-9.]* *memory: *[0-9.]*GiB" $OUT/$nm.log | sed 's/^/  /' >> $R
  grep -m3 -E "Error|Traceback" $OUT/$nm.log | sed 's/^/  ! /' >> $R; }
OFF=--parallelism.no-enable-sequence-parallel
run warm_after_spoff  $AFTER 1 $OFF; run warm_before_spoff $BEFORE 1 $OFF
run warm_after_spon   $AFTER 1;     run warm_before_spon  $BEFORE 1
run before_spoff $BEFORE 3 $OFF; run after_spoff $AFTER 3 $OFF; run after_spoff_again $AFTER 3 $OFF
run before_spon  $BEFORE 3;      run after_spon  $AFTER 3
echo "DONE $OUT" >> $R; echo "DONE $OUT"
