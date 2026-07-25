#!/bin/bash
# packed-MXFP4 QLoRA x TP GPU matrix (2026-07-25, 8x5060Ti box #2).
# SESSION_HANDOFF_2026-07-24 sec 7 item 1: the cells that could not run on
# the box that died. Fork ef0fced4 wired TP for the packed base; only the
# CPU-gloo probe (packed_tp_cpu_probe.py) had run.
#
# Prereq -- rebuild the offline quantize-then-shard artifacts:
#   1. bf16 gated_lora run, --checkpoint.enable --checkpoint.interval 1
#      (--dump-folder $BF16_OUT), 1 step
#   2. stream_quantize_mxfp4_dcp.py --src $BF16_OUT/checkpoint/step-1 \
#        --dst $PACKED
# The packed base values therefore depend on the bf16 source run; absolute
# step-1 losses are only comparable WITHIN one $PACKED artifact (see the
# report's note on the 07-24 handoff's 7.5695).
#
# GLOBAL_BATCH is pinned on every cell: global_batch_size defaults to
# local_batch_size * dp_degree, so a dp2 cell would otherwise optimize a
# 2x larger batch than a dp1 cell and its loss would not be comparable.
# The 1-GPU reference is the anchor for the dp1 TP cells.
#
# Gate per cell: descending finite loss, step-1 within TP's numeric band of
# the reference, AND rank-identical loss/grad_norm (Part-3 lesson: "loss
# descends in a band" is not a multi-rank correctness gate).
set -u
cd /workspace/torchtitan_attention_residual/torchtitan
PACKED=${PACKED:-/workspace/packed_mxfp4_ckpt}
STEPS=${STEPS:-5}
GLOBAL_BATCH=${GLOBAL_BATCH:-4}
CFG="--module kimi_k3 --config kimi_linear_debugmodel_gated_qlora_mxfp4 \
 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
 --training.global-batch-size $GLOBAL_BATCH \
 --checkpoint.enable --checkpoint.initial-load-path $PACKED \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"
PORT=29900

run_cell() {
  local name="$1" ngpu="$2"; shift 2
  PORT=$((PORT+1))
  echo "=================== CELL: $name (${ngpu} ranks) ==================="
  local out clean lines
  out=$(CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) torchrun --nproc_per_node=$ngpu \
        --master_port=$PORT -m torchtitan.train $CFG --training.steps $STEPS \
        --dump-folder /workspace/out_qlora_tp/$name "$@" 2>&1)
  clean=$(echo "$out" | sed -E 's/\x1b\[[0-9;]*m//g')
  # every rank logs; unique (step, loss, grad_norm) triples must number
  # exactly STEPS if all ranks agree -- more lines means rank divergence.
  # -4.00000 / -2.00000 are the PP sentinel losses logged by stages that
  # do not own the loss; dropping them (as run_cp_tp_3d_matrix.sh does)
  # keeps the rank-identical count meaningful for PP cells.
  lines=$(echo "$clean" | grep -E "step: +[0-9]+ +loss" \
    | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+  grad_norm: +[-0-9.]+).*/\1/' \
    | grep -vE '\-4\.00000|\-2\.00000' | sort -u)
  echo "$lines"
  echo "-- unique step/loss/grad_norm lines: $(echo "$lines" | grep -c .) (expect $STEPS if rank-identical)"
  echo "$clean" | grep -oE "memory: +[0-9.]+GiB" | tail -1
  echo "$clean" | grep -iE "traceback|RuntimeError|ValueError|NotImplementedError|AssertionError" | head -3
}

# reference: no parallelism at all, same global batch as the dp1 TP cells
run_cell "ref_1gpu" 1

# the three cells sec 7 item 1 asks for (all dp1, so directly vs ref_1gpu)
run_cell "tp2"     2 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2
run_cell "tp2cp2"  4 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2
run_cell "tp2pp2"  4 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B

# the handoff's quoted baseline shape (FSDP2) + the FSDP x TP x CP composition
run_cell "fsdp2"       2 --parallelism.data_parallel_shard_degree 2
run_cell "fsdp2tp2cp2" 8 --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2

echo "=================== PACKED-TP MATRIX DONE ==================="
