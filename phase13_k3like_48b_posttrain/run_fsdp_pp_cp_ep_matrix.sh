#!/bin/bash
# FSDP x PP x CP (choose 1, 2, 3) x EP on/off, all from ONE seed checkpoint.
#
# TP is deliberately excluded: it carries an unattributed one-directional
# gradient gap (TP_GRAD_FINDING_2026-07-29), so including it would mix a known
# open question into a matrix meant to establish the other axes.
#
# Sharing a seed checkpoint is what makes the losses comparable at all. Without
# it FSDP2's meta-init gives every rank its own RNG stream, so two parallel
# degrees start from different weights and only "it runs, rank-identically" can
# be claimed. With it, the numbers can be compared to each other.
#
# EP is carved out of the data-parallel axes, so an EP leg needs dp_shard >= ep.
# Legs where that is impossible (pp-only, cp-only, pp+cp -- all dp_shard=1) are
# reported as STRUCTURALLY N/A rather than silently skipped.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/workspace/matrix_fpce}
SEED=$OUT/seed
STEPS=3
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=kimi_k3_mini_block_attn_res
BASE="--module kimi_k3 --config $FLAVOR --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1 --training.steps $STEPS"

losses() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+ +grad_norm: +[-0-9.]+).*/\1/' \
  | grep -vE 'loss: +-'; }
fails() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -iE \
  "traceback|RuntimeError|ValueError|AssertionError|KeyError|NotImplementedError" | head -2; }

echo "########## seed checkpoint ##########"
rm -rf "$SEED"
CUDA_VISIBLE_DEVICES=0 timeout 1800 torchrun --nproc_per_node=1 --master_port=34000 \
  -m torchtitan.train $BASE --training.steps 1 --training.global-batch-size 1 \
  --training.local-batch-size 1 --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.enable --checkpoint.create-seed-checkpoint --dump-folder "$SEED" 2>&1 | tail -2
SEED_PATH=$(find "$SEED" -maxdepth 3 -type d -name "step-0" | head -1)
echo "seed: ${SEED_PATH:-MISSING}"
[ -z "${SEED_PATH:-}" ] && { echo "ABORT: no seed"; exit 1; }
LOAD="--checkpoint.enable --checkpoint.initial-load-path $SEED_PATH \
 --checkpoint.initial-load-model-only --checkpoint.interval 100000"

PORT=34100
run() {
  local name="$1" ngpu="$2"; shift 2
  PORT=$((PORT+1))
  echo "=== $name (${ngpu} GPU) ==="
  local out uniq n_uniq n_total
  out=$(CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) timeout 2400 torchrun \
        --nproc_per_node=$ngpu --master_port=$PORT -m torchtitan.train \
        $BASE $LOAD "$@" --dump-folder "$OUT/$name" 2>&1)
  uniq=$(echo "$out" | losses | sort -u); echo "$uniq"
  n_uniq=$(echo "$uniq" | grep -c "step:"); n_total=$(echo "$out" | losses | grep -c "step:")
  if [ "$n_total" -eq 0 ]; then echo "  -> FAIL (no steps)"; echo "$out" | fails
  elif [ "$n_uniq" -eq "$STEPS" ]; then echo "  -> PASS rank-identical"
  else echo "  -> FAIL rank divergence ($n_uniq distinct, expected $STEPS)"; fi
}
GB8="--training.global-batch-size 8"
PPB="--training.local-batch-size 2"   # PP needs microbatches >= stages

echo; echo "########## choose 1 ##########"
run fsdp2            2 $GB8 --parallelism.data_parallel_shard_degree 2
run pp2              2 $GB8 $PPB --parallelism.data_parallel_shard_degree 1 \
                                 --parallelism.pipeline_parallel_degree 2
run cp2              2 $GB8 --parallelism.data_parallel_shard_degree 1 \
                            --parallelism.context_parallel_degree 2

echo; echo "########## choose 2 ##########"
run fsdp2_pp2        4 $GB8 $PPB --parallelism.data_parallel_shard_degree 2 \
                                 --parallelism.pipeline_parallel_degree 2
run fsdp2_cp2        4 $GB8 --parallelism.data_parallel_shard_degree 2 \
                            --parallelism.context_parallel_degree 2
run pp2_cp2          4 $GB8 $PPB --parallelism.data_parallel_shard_degree 1 \
                                 --parallelism.pipeline_parallel_degree 2 \
                                 --parallelism.context_parallel_degree 2

echo; echo "########## all three ##########"
run fsdp2_pp2_cp2    8 $GB8 $PPB --parallelism.data_parallel_shard_degree 2 \
                                 --parallelism.pipeline_parallel_degree 2 \
                                 --parallelism.context_parallel_degree 2

echo; echo "########## EP on (needs dp_shard >= ep) ##########"
run ep2_fsdp2        2 $GB8 --parallelism.data_parallel_shard_degree 2 \
                            --parallelism.expert_parallel_degree 2
run ep2_fsdp4        4 $GB8 --parallelism.data_parallel_shard_degree 4 \
                            --parallelism.expert_parallel_degree 2
run ep2_fsdp2_pp2    4 $GB8 $PPB --parallelism.data_parallel_shard_degree 2 \
                                 --parallelism.expert_parallel_degree 2 \
                                 --parallelism.pipeline_parallel_degree 2
run ep2_fsdp2_cp2    4 $GB8 --parallelism.data_parallel_shard_degree 2 \
                            --parallelism.expert_parallel_degree 2 \
                            --parallelism.context_parallel_degree 2
run ep2_fsdp2_pp2_cp2 8 $GB8 $PPB --parallelism.data_parallel_shard_degree 2 \
                                  --parallelism.expert_parallel_degree 2 \
                                  --parallelism.pipeline_parallel_degree 2 \
                                  --parallelism.context_parallel_degree 2
echo
echo "STRUCTURALLY N/A with EP: pp-only, cp-only, pp+cp -- all need dp_shard=1,"
echo "and EP is carved out of the data-parallel axes, so ep2 cannot fit."
echo "########## DONE ##########"
