#!/bin/bash
# Corrected legs of run_remaining_8gpu_gaps.sh (2026-07-25):
#   sec 1: EP is carved out of the data-parallel axes, so deepseek_v3 on 4
#          ranks with tp2+ep2 needs dp_shard=2 (dp2*tp2=4), not dp_shard=1.
#   sec 2: the DCP legs needed --checkpoint.interval 2 to actually produce a
#          mid-run step-2 save; $LOAD's interval 100000 was not overridden, so
#          no step-2 existed and the cross-mesh legs had nothing to load.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
PACKED=${PACKED:-/workspace/packed_mxfp4_ckpt}
OUT=${OUT:-/workspace/out_redo}
cd "$TITAN"
export PYTHONPATH=$TITAN
QCFG="--module kimi_k3 --config kimi_linear_debugmodel_gated_qlora_mxfp4 \
 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
 --training.global-batch-size 4"
PACKED_LOAD="--checkpoint.enable --checkpoint.initial-load-path $PACKED \
 --checkpoint.initial-load-model-only"

losses() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+  grad_norm: +[-0-9.]+).*/\1/' \
  | grep -vE '\-4\.00000|\-2\.00000' | sort -u; }
fails() { sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -iE "traceback|RuntimeError|ValueError|NotImplementedError|AssertionError" | head -3; }
run() { local name="$1" ngpu="$2" port="$3"; shift 3
  echo "=== $name ==="
  local out
  out=$(CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) torchrun --nproc_per_node=$ngpu \
        --master_port=$port -m torchtitan.train "$@" 2>&1)
  echo "$out" | losses; echo "$out" | fails
}

echo "############ 1. PR16 evidence: common-MoE TP+EP scatter, UPSTREAM model ############"
DS3="--module deepseek_v3 --config deepseek_v3_debugmodel --checkpoint.no-enable \
 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 2 \
 --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 \
 --parallelism.expert_parallel_degree 2"
run "deepseek_v3 dp2xtp2xep2 WITH our moe fix" 4 30401 $DS3 --dump-folder $OUT/ds3_with_fix
if git show 129e29de -- torchtitan/models/common/moe.py | git apply -R 2>/dev/null; then
  echo "-- moe.py scatter fix REVERTED (= upstream main state) --"
  run "deepseek_v3 dp2xtp2xep2 WITHOUT the fix" 4 30402 $DS3 --dump-folder $OUT/ds3_no_fix
  git checkout -- torchtitan/models/common/moe.py
  echo "-- fix restored --"
else
  echo "REVERSE PATCH FAILED -- without-fix leg NOT run (not silently dropped)"
fi

echo "############ 2. packed-MXFP4 DCP under TP (corrected) ############"
TP2="--parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2"
run "2a tp2 reference, 4 steps, no save" 2 30411 $QCFG $PACKED_LOAD \
  --checkpoint.interval 100000 --training.steps 4 $TP2 --dump-folder $OUT/dcp_ref
run "2b tp2 save, interval 2 -> step-2 is a mid-run FULL save" 2 30412 $QCFG $PACKED_LOAD \
  --checkpoint.interval 2 --training.steps 3 $TP2 --dump-folder $OUT/dcp_save
ls -d $OUT/dcp_save/checkpoint/step-* 2>/dev/null || echo "NO CHECKPOINTS WRITTEN"
# isolate step-2 so auto-resume cannot pick the step-3 model-only export
rm -rf $OUT/dcp_resume; mkdir -p $OUT/dcp_resume/checkpoint
cp -r $OUT/dcp_save/checkpoint/step-2 $OUT/dcp_resume/checkpoint/step-2 || echo "COPY FAILED"
run "2c tp2 same-mesh resume from step-2 -> steps 3-4 must match 2a" 2 30413 $QCFG \
  --checkpoint.enable --checkpoint.interval 100000 --training.steps 4 $TP2 \
  --dump-folder $OUT/dcp_resume
# cross-mesh reshard: identical weights + identical (restarted) data on three
# meshes -> the three step-1 losses must agree within the TP band.
XLOAD="--checkpoint.enable --checkpoint.initial-load-path $OUT/dcp_save/checkpoint/step-2 \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000 \
 --checkpoint.exclude-from-loading dataloader,lr_scheduler,optimizer,train_state"
run "2d cross-mesh -> tp2 (control)" 2 30414 $QCFG $XLOAD --training.steps 1 $TP2 \
  --dump-folder $OUT/x_tp2
run "2e cross-mesh -> tp1" 1 30415 $QCFG $XLOAD --training.steps 1 --dump-folder $OUT/x_tp1
run "2f cross-mesh -> fsdp2" 2 30416 $QCFG $XLOAD --training.steps 1 \
  --parallelism.data_parallel_shard_degree 2 --dump-folder $OUT/x_fsdp2
run "2g cross-mesh -> tp2cp2 (4 ranks)" 4 30417 $QCFG $XLOAD --training.steps 1 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 \
  --parallelism.context_parallel_degree 2 --dump-folder $OUT/x_tp2cp2

echo "############ GAPS REDO DONE ############"

echo "############ 3. compile x packed-TP (FULL output -- the sweep run printed no steps) ############"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=30421 -m torchtitan.train \
  $QCFG $PACKED_LOAD --checkpoint.interval 100000 --training.steps 3 $TP2 \
  --compile.enable --dump-folder $OUT/compile_tp2 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -viE "^\[titan\].*WARNING|_inductor/codecache|_pytree" | tail -25
echo "############ REDO PART 2 DONE ############"
