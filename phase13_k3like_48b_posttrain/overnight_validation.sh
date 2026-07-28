#!/bin/bash
# Overnight validation: everything this box can still verify (2026-07-28).
#
# Ordered so the expensive, highest-value gap goes first, and so later phases
# reuse what earlier ones produce (the seed checkpoint unblocks both the
# cross-parallelism comparison AND veRL's actor load).
#
# Every phase logs PASS/FAIL and continues on failure -- one broken phase must
# not hide the others. GPU work is serialized; nothing runs concurrently.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
PHASE13=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain
OUT=${OUT:-/workspace/overnight}
SEED=$OUT/seed_ckpt
mkdir -p "$OUT"
cd "$TITAN"
export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=kimi_linear_k3mini_block_attn_res
COMMON="--module kimi_k3 --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1"

say() { echo; echo "########## $* ##########"; }
losses() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+ +grad_norm: +[-0-9.]+).*/\1/' \
  | grep -vE 'loss: +-'; }
fails() { sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -iE "traceback|RuntimeError|ValueError|NotImplementedError|AssertionError|KeyError" \
  | head -3; }

# run <name> <ngpu> <port> <extra args...>
run() {
  local name="$1" ngpu="$2" port="$3"; shift 3
  echo "=== $name (${ngpu} GPU) ==="
  local out uniq n_uniq n_total
  out=$(CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) timeout 2400 torchrun \
        --nproc_per_node=$ngpu --master_port=$port -m torchtitan.train \
        $COMMON "$@" --dump-folder "$OUT/$name" 2>&1)
  uniq=$(echo "$out" | losses | sort -u); echo "$uniq"
  n_uniq=$(echo "$uniq" | grep -c "step:"); n_total=$(echo "$out" | losses | grep -c "step:")
  if [ "$n_total" -eq 0 ]; then echo "  -> FAIL (no steps)"; echo "$out" | fails
  elif [ "$n_uniq" -eq "$STEPS" ]; then echo "  -> PASS rank-identical"
  else echo "  -> FAIL rank divergence ($n_uniq distinct, expected $STEPS)"; fi
}

############################################################################
say "PHASE 1: seed checkpoint (the missing correctness gate)"
# Cross-parallelism loss EQUALITY was the one validation this repo never had:
# FSDP2 meta-init gives every rank its own RNG stream, so two parallel degrees
# start from different weights and their curves cannot be compared. A seed
# checkpoint fixes the init, which turns "it composes" into "it computes the
# same thing".
STEPS=1
rm -rf "$SEED"
CUDA_VISIBLE_DEVICES=0 timeout 2400 torchrun --nproc_per_node=1 --master_port=31001 \
  -m torchtitan.train $COMMON --config $FLAVOR --training.steps 1 \
  --training.global-batch-size 1 --training.local-batch-size 1 \
  --parallelism.data_parallel_shard_degree 1 \
  --checkpoint.enable --checkpoint.create-seed-checkpoint \
  --dump-folder "$SEED" 2>&1 | tail -5
find "$SEED" -maxdepth 3 -name "*.distcp" 2>/dev/null | head -3
SEED_PATH=$(find "$SEED" -maxdepth 3 -type d -name "step-0" | head -1)
echo "seed checkpoint: ${SEED_PATH:-NOT FOUND}"

############################################################################
say "PHASE 2: cross-parallelism loss equality from a shared init"
STEPS=3
if [ -n "${SEED_PATH:-}" ]; then
  LOAD="--checkpoint.enable --checkpoint.initial-load-path $SEED_PATH \
        --checkpoint.initial-load-model-only --checkpoint.interval 100000"
  GB="--training.global-batch-size 8"
  run seed_dp8       8 31101 --config $FLAVOR --training.steps 3 $GB $LOAD \
      --parallelism.data_parallel_shard_degree 8
  run seed_dp2       2 31102 --config $FLAVOR --training.steps 3 $GB $LOAD \
      --parallelism.data_parallel_shard_degree 2
  run seed_dp2xtp2   4 31103 --config $FLAVOR --training.steps 3 $GB $LOAD \
      --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2
  run seed_dp2xcp2   4 31104 --config $FLAVOR --training.steps 3 $GB $LOAD \
      --parallelism.data_parallel_shard_degree 2 --parallelism.context_parallel_degree 2
  run seed_dp2xpp2   4 31105 --config $FLAVOR --training.steps 3 $GB $LOAD \
      --training.local-batch-size 2 \
      --parallelism.data_parallel_shard_degree 2 --parallelism.pipeline_parallel_degree 2
  run seed_dp2xep2   4 31106 --config $FLAVOR --training.steps 3 $GB $LOAD \
      --parallelism.data_parallel_shard_degree 4 --parallelism.expert_parallel_degree 2
  run seed_3d        8 31107 --config $FLAVOR --training.steps 3 $GB $LOAD \
      --parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 \
      --parallelism.context_parallel_degree 2
else
  echo "  -> SKIP (no seed checkpoint)"
fi

############################################################################
say "PHASE 3: every K3 flavor from the same seed"
STEPS=3
PORT_OFF=0
for f in kimi_linear_k3mini_qat_mxfp4 kimi_linear_k3mini_qlora \
         kimi_linear_k3mini_kcp kimi_linear_k3mini_quantile_balance; do
  extra=""
  [ "$f" = "kimi_linear_k3mini_kcp" ] && extra="--training.local-batch-size 1 \
    --parallelism.context_parallel_degree 2 --parallelism.data_parallel_shard_degree 1"
  [ -z "$extra" ] && extra="--parallelism.data_parallel_shard_degree 2"
  PORT=$((31200 + PORT_OFF)); PORT_OFF=$((PORT_OFF + 1))
  run "flavor_${f#kimi_linear_k3mini_}" 2 $PORT --config $f \
      --training.steps 3 --training.global-batch-size 2 $extra
done

############################################################################
say "PHASE 4: multimodal at dp2 (vision tower + backbone together)"
PYTHONPATH=$TITAN timeout 1200 python -m pytest \
  torchtitan/experiments/kimi_k3/tests/test_k3_multimodal.py \
  torchtitan/experiments/kimi_k3/tests/test_moonvit.py -q 2>&1 | tail -3
PYTHONPATH=$TITAN timeout 1200 torchrun --nproc_per_node=2 --master_port=31401 \
  "$PHASE13/vision_fsdp_probe.py" 2>&1 | grep -E "VIS-FSDP" | tail -6

############################################################################
say "PHASE 5: CP numerics re-verified (per-layer, fp32)"
for n in 2 4; do
  PYTHONPATH=$TITAN timeout 1800 torchrun --nproc_per_node=$n --master_port=3150$n \
    "$PHASE13/mixed_cp_parity_probe.py" 512 2>&1 | grep -E "MIXED-CP|Error" | tail -6
done
PYTHONPATH=$TITAN timeout 1200 torchrun --nproc_per_node=2 --master_port=31521 \
  "$PHASE13/conv_halo_probe.py" 512 2>&1 | grep -E "CONV-CP" | tail -4
PYTHONPATH=$TITAN timeout 1200 torchrun --nproc_per_node=2 --master_port=31522 \
  "$PHASE13/kda_kcp_module_probe.py" 512 2>&1 | grep -E "KCP-MOD" | tail -5

############################################################################
say "PHASE 6: MoE connectivity + expert-init guard (the regression that cost the baselines)"
PYTHONPATH=$TITAN timeout 1200 torchrun --nproc_per_node=2 --master_port=31601 \
  "$PHASE13/moe_connected_probe.py" 2>&1 | grep -E "\[MOE\]" | tail -6

############################################################################
say "PHASE 7: veRL actor from the seed checkpoint"
# The engine previously stopped at "Missing key in checkpoint" because
# /workspace/k3mini_hf has a config but no weights. A seed checkpoint is a real
# one, so this is the first chance the actor path has had to load anything.
if [ -n "${SEED_PATH:-}" ]; then
  PYTHONPATH=$TITAN VERL_SEED_CKPT="$SEED_PATH" timeout 1800 torchrun \
    --nproc_per_node=1 --master_port=31701 "$PHASE13/verl_actor_smoke.py" 2>&1 \
    | grep -E "\[VERL\]|Error|error" | tail -8
else
  echo "  -> SKIP (no seed checkpoint)"
fi

############################################################################
say "PHASE 8: full unit suite (regression gate)"
timeout 2400 python -m pytest torchtitan/experiments/kimi_k3/tests/ -q 2>&1 | tail -4

say "DONE"
