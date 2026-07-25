#!/bin/bash
# Everything still open in the 07-24/07-25 handoffs that FITS ON 8x16GB
# (2026-07-25, box #2). Runs after the post-grad-fix regression matrices.
#
#   1. PR16 evidence: the common-MoE TP+EP scatter fix reproduced on an
#      UPSTREAM model (deepseek_v3 debugmodel), with and without the fix.
#   2. DCP for the PACKED-MXFP4 checkpoint under TP: mid-run save,
#      same-mesh resume, and cross-mesh reshard parity (tp2 ckpt -> tp1 /
#      fsdp2). ef0fced4 claimed the DTensor registration keeps this
#      working; it was never run.
#   3. AC full x packed-TP (AC must not change forward numerics).
#   4. compile x packed-TP.
#   5. GAPS sec 5 staleness check: bf16 gated_lora at dp1+tp2 (no FSDP).
#   6. B7 Muon x FSDP x TP x CP capstone, re-run post grad-division fix.
#   7. B8 deltas-compose capstone (gated MLA + graft + QAT + Muon).
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
PHASE13=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain
PACKED=${PACKED:-/workspace/packed_mxfp4_ckpt}
OUT=${OUT:-/workspace/out_gaps}
cd "$TITAN"
export PYTHONPATH=$TITAN
QCFG="--module kimi_k3 --config kimi_linear_debugmodel_gated_qlora_mxfp4 \
 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
 --training.global-batch-size 4"
LOAD="--checkpoint.enable --checkpoint.initial-load-path $PACKED \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

losses() {  # strip ANSI, unique (step, loss, grad_norm), drop PP sentinels
  sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+  grad_norm: +[-0-9.]+).*/\1/' \
  | grep -vE '\-4\.00000|\-2\.00000' | sort -u
}
fails() { sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -iE "traceback|RuntimeError|ValueError|NotImplementedError|AssertionError" | head -3; }
run() { local name="$1" ngpu="$2" port="$3"; shift 3
  echo "=== $name ==="
  local out
  out=$(CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) torchrun --nproc_per_node=$ngpu \
        --master_port=$port -m torchtitan.train "$@" 2>&1)
  echo "$out" | losses; echo "$out" | fails
}

echo "############ 1. PR16: common-MoE TP+EP scatter on an UPSTREAM model ############"
DS3="--module deepseek_v3 --config deepseek_v3_debugmodel --checkpoint.no-enable \
 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 2 \
 --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 \
 --parallelism.expert_parallel_degree 2"
run "deepseek_v3 tp2xep2 WITH our moe fix" 4 30301 $DS3 --dump-folder $OUT/ds3_with_fix
if git show 129e29de -- torchtitan/models/common/moe.py | git apply -R 2>/dev/null; then
  echo "-- moe.py scatter fix REVERTED (upstream state) --"
  run "deepseek_v3 tp2xep2 WITHOUT the fix (= upstream main)" 4 30302 $DS3 --dump-folder $OUT/ds3_no_fix
  git checkout -- torchtitan/models/common/moe.py
  echo "-- fix restored --"
else
  echo "REVERSE PATCH FAILED -- skipped the without-fix leg (nothing silently dropped)"
fi

echo "############ 2. packed-MXFP4 DCP under TP ############"
# 2a reference trajectory, no checkpointing at all
run "2a tp2 reference, 4 steps, no ckpt" 2 30311 $QCFG $LOAD --training.steps 4 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 \
  --dump-folder $OUT/dcp_ref
# 2b mid-run save: steps 3 with interval 2 -> step-2 is a FULL save (not the
# model-only last-step export), step-3 is the last-step export.
run "2b tp2 save (step-2 full save)" 2 30312 $QCFG $LOAD --training.steps 3 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 \
  --dump-folder $OUT/dcp_save
# copy ONLY step-2 into a fresh folder so auto-resume cannot pick step-3
rm -rf $OUT/dcp_resume; mkdir -p $OUT/dcp_resume/checkpoint
cp -r $OUT/dcp_save/checkpoint/step-2 $OUT/dcp_resume/checkpoint/step-2
run "2c tp2 same-mesh resume from step-2 -> steps 3-4 must match 2a" 2 30313 $QCFG \
  --training.steps 4 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 \
  --checkpoint.enable --checkpoint.interval 100000 --dump-folder $OUT/dcp_resume
# cross-mesh reshard parity: same weights + same (restarted) data, 3 meshes.
# All three must agree within the TP band; that is the reshard gate.
XLOAD="--checkpoint.enable --checkpoint.initial-load-path $OUT/dcp_save/checkpoint/step-2 \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000 \
 --checkpoint.exclude-from-loading dataloader,lr_scheduler,optimizer,train_state"
run "2d cross-mesh tp2 ckpt -> tp2 (control)" 2 30314 $QCFG $XLOAD --training.steps 1 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 \
  --dump-folder $OUT/x_tp2
run "2e cross-mesh tp2 ckpt -> tp1" 1 30315 $QCFG $XLOAD --training.steps 1 \
  --dump-folder $OUT/x_tp1
run "2f cross-mesh tp2 ckpt -> fsdp2" 2 30316 $QCFG $XLOAD --training.steps 1 \
  --parallelism.data_parallel_shard_degree 2 --dump-folder $OUT/x_fsdp2

echo "############ 3. AC full x packed-TP (must equal non-AC tp2) ############"
run "AC full x tp2" 2 30321 $QCFG $LOAD --training.steps 3 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 \
  --dump-folder $OUT/ac_tp2 activation-checkpoint:full

echo "############ 4. compile x packed-TP ############"
run "compile x tp2" 2 30331 $QCFG $LOAD --training.steps 3 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 \
  --compile.enable --dump-folder $OUT/compile_tp2

echo "############ 5. GAPS sec 5 staleness: bf16 gated_lora dp1+tp2 (no FSDP) ############"
run "gated_lora dp1 tp2 (no FSDP)" 2 30341 --module kimi_k3 \
  --config kimi_linear_debugmodel_gated_lora --checkpoint.no-enable \
  --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
  --training.global-batch-size 4 --training.steps 3 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 \
  --dump-folder $OUT/lora_dp1tp2

echo "############ 6. B7 Muon x FSDP x TP x CP capstone (post grad-fix) ############"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=30351 \
  "$PHASE13/muon_tp_cp_capstone.py" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -iE "loss|PASS|FAIL|error" | tail -8

echo "############ 7. B8 deltas-compose capstone (post grad-fix) ############"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=30361 \
  "$PHASE13/mgpu_capstone_deltas.py" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -iE "loss|PASS|FAIL|error" | tail -8

echo "############ REMAINING-GAPS SWEEP DONE ############"
