#!/usr/bin/env bash
# Phase 6 / Phase 7 generic 8-GPU multimodal launcher.
#
# Parameterized by mesh + recipe env vars so the same script drives:
#   - alignment runs (Tier C, GBS=12, 500 steps)
#   - production-realistic traces (Tier B, GBS=120, 50 steps)
#   - production-standardized traces (Tier A, GBS=384, 100 steps)
#   - v10 continued pretrain (GBS=120, 5000 steps)
#
# Required env (no defaults — caller MUST set):
#   OUT_DIR         per-run output dir (containing tb/, train.log, ckpt/)
#   FSDP, PP, TP, CP, EP   parallelism degrees (must multiply to 8 except
#                          when running on fewer ranks via NGPU)
#   STEPS, LOCAL_BS, GLOBAL_BS, SEQ_LEN
#
# Optional env:
#   NGPU=8                    total ranks (must equal FSDP*PP*TP*CP*EP)
#   V=2                       virtual stages per PP rank (only if PP>1)
#   ADAPTER=1                 TORCHTITAN_ATTNRES_CACHE=1 (only if PP>1)
#   FLAVOR=kimi_linear_436m_block_attn_res_n4
#   STUDENT_CKPT=$WORKSPACE/phase4/runs/.../step-8000
#   SEED=42
#   DETERMINISTIC=1
#   COMPILE=1
#   TRACE_TIER=                empty=no trace; tier_a/b/c/d enables NCCL trace capture
#   TRACE_STEPS=50             how many steps to profile (skip first 10, then warmup 5, active=TRACE_STEPS-15)
#   PROJ_LR_MULT=50.0
#   LR=1e-5
#   WARMUP=10
#   LOG_FREQ=1                 metrics.log_freq for tb dump
#   SAVE_FREQ=999999           checkpoint.interval; default = effectively never
#   KEEP_K=2

set -euo pipefail

if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

# ---- required ----
: "${OUT_DIR:?OUT_DIR is required}"
: "${FSDP:?FSDP degree required}"
: "${PP:?PP degree required}"
: "${TP:?TP degree required}"
: "${CP:?CP degree required}"
: "${EP:?EP degree required}"
: "${STEPS:?STEPS required}"
: "${LOCAL_BS:?LOCAL_BS required}"
: "${GLOBAL_BS:?GLOBAL_BS required}"

# ---- defaults ----
NGPU="${NGPU:-8}"
V="${V:-1}"
ADAPTER="${ADAPTER:-1}"
FLAVOR="${FLAVOR:-kimi_linear_436m_block_attn_res_n4}"
STUDENT_CKPT="${STUDENT_CKPT:-${WORKSPACE_DIR}/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000}"
SEED="${SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-1}"
COMPILE="${COMPILE:-1}"
TRACE_TIER="${TRACE_TIER:-}"
TRACE_STEPS="${TRACE_STEPS:-50}"
PROJ_LR_MULT="${PROJ_LR_MULT:-50.0}"
LR="${LR:-1e-5}"
WARMUP="${WARMUP:-10}"
LOG_FREQ="${LOG_FREQ:-1}"
SAVE_FREQ="${SAVE_FREQ:-999999}"
KEEP_K="${KEEP_K:-2}"
SEQ_LEN="${SEQ_LEN:-260}"
MM_GLOBAL_SEQ_LEN="${MM_GLOBAL_SEQ_LEN:-258}"

# ---- data paths (8-GPU box layout) ----
DATA_DIR="${DATA_DIR:-/workspace/.hf_home/LLaVA-Pretrain}"
JSON="${JSON:-${DATA_DIR}/blip_laion_cc_sbu_558k.json}"
IMAGES="${IMAGES:-${DATA_DIR}}"
VISION="${VISION:-google/siglip-base-patch16-224}"
TOKENIZER="${TOKENIZER:-NousResearch/Meta-Llama-3.1-8B}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/workspace/.hf_home}"

# ---- arithmetic sanity ----
# In torchtitan, EP borrows from existing FSDP+TP axes via the sparse mesh
# unflatten, so the dense-mesh product is FSDP*PP*TP*CP and must equal NGPU.
# The constraint EP*ETP <= FSDP*TP must also hold (EP=1 trivially satisfies).
DENSE_PRODUCT=$(( FSDP * PP * TP * CP ))
if [[ "$DENSE_PRODUCT" != "$NGPU" ]]; then
    echo "ERROR: dense FSDP($FSDP) * PP($PP) * TP($TP) * CP($CP) = $DENSE_PRODUCT != NGPU($NGPU)" >&2
    echo "       (EP=$EP borrows from FSDP×TP and does not enter the dense product)" >&2
    exit 1
fi
if [[ "$EP" -gt 1 ]]; then
    EP_BUDGET=$(( FSDP * TP ))
    if [[ "$EP" -gt "$EP_BUDGET" ]]; then
        echo "ERROR: EP($EP) > FSDP*TP($EP_BUDGET); EP must borrow from existing dense axes" >&2
        exit 1
    fi
fi

# ---- mesh-derived flags ----
COMPILE_ARG=""
if [[ "$COMPILE" == "1" ]]; then
    COMPILE_ARG="--compile.enable"
fi

CKPT_ARGS=""
CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-0}"
if [[ "$CHECKPOINT_ENABLED" == "1" && -d "$STUDENT_CKPT" ]]; then
    # Long pretrain: rolling checkpoints, keep latest K=KEEP_K (default 2)
    CKPT_ARGS="--checkpoint.enable --checkpoint.initial_load_path ${STUDENT_CKPT} --checkpoint.initial_load_model_only --checkpoint.interval ${SAVE_FREQ} --checkpoint.keep_latest_k ${KEEP_K}"
elif [[ -d "$STUDENT_CKPT" ]]; then
    # Alignment / trace tier: load init only, never save (interval set
    # ridiculously high). See phase6/CHECKPOINT_RULES.md for the rule.
    CKPT_ARGS="--checkpoint.enable --checkpoint.initial_load_path ${STUDENT_CKPT} --checkpoint.initial_load_model_only --checkpoint.interval 999999999 --checkpoint.keep_latest_k 1"
fi

DEBUG_ARGS=""
if [[ -n "$SEED" ]]; then
    DEBUG_ARGS="$DEBUG_ARGS --debug.seed $SEED"
fi
if [[ "$DETERMINISTIC" == "1" ]]; then
    DEBUG_ARGS="$DEBUG_ARGS --debug.deterministic"
fi

# PP schedule + cache adapter only when PP > 1
PP_ARGS="--parallelism.pipeline_parallel_degree $PP"
if [[ "$PP" -gt 1 ]]; then
    PP_ARGS="$PP_ARGS --parallelism.pipeline_parallel_schedule Interleaved1F1B"
    PP_ARGS="$PP_ARGS --parallelism.pipeline_parallel_layers_per_stage $V"
    PP_ARGS="$PP_ARGS --parallelism.pipeline_parallel_first_stage_less_layers 0"
    PP_ARGS="$PP_ARGS --parallelism.pipeline_parallel_last_stage_less_layers 0"
    if [[ "$ADAPTER" == "1" ]]; then
        export TORCHTITAN_ATTNRES_CACHE=1
    else
        unset TORCHTITAN_ATTNRES_CACHE
    fi
fi

# torchtitan dense_mesh layout is (pp, dp_replicate, fsdp, tp) outer→inner,
# so rank index = pp*(dp_rep*fsdp*tp) + dp_rep*(fsdp*tp) + fsdp*tp + tp_idx.
# Last PP stage starts at rank (PP-1) * dp_replicate * fsdp * tp.
# We default LOG_RANK there (loss-bearing rank).
DP_REP=1
LAST_PP_RANK_BASE="$(( (PP - 1) * DP_REP * FSDP * TP ))"
LOG_RANK="${LOG_RANK:-${LAST_PP_RANK_BASE}}"

# ---- trace capture wrapper ----
TRACE_ENV=""
if [[ -n "$TRACE_TIER" ]]; then
    TRACE_DIR="${OUT_DIR}/${TRACE_TIER}_trace"
    mkdir -p "$TRACE_DIR"
    TRACE_ENV="\
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=COLL,INIT \
NCCL_DEBUG_FILE=${TRACE_DIR}/nccl-rank-%h-%p.log \
TORCH_NCCL_TRACE_BUFFER_SIZE=20000 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_USE_COMM_NONBLOCKING=1 \
PHASE7_PROFILE_DIR=${TRACE_DIR} \
PHASE7_PROFILE_STEPS=${TRACE_STEPS}"
fi

mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/recipe.json" <<EOF
{
  "out_dir": "$OUT_DIR",
  "ngpu": $NGPU,
  "fsdp": $FSDP,
  "pp": $PP,
  "tp": $TP,
  "cp": $CP,
  "ep": $EP,
  "v": $V,
  "adapter": $ADAPTER,
  "flavor": "$FLAVOR",
  "student_ckpt": "$STUDENT_CKPT",
  "seed": "$SEED",
  "deterministic": $DETERMINISTIC,
  "compile": $COMPILE,
  "trace_tier": "$TRACE_TIER",
  "steps": $STEPS,
  "local_bs": $LOCAL_BS,
  "global_bs": $GLOBAL_BS,
  "seq_len": $SEQ_LEN,
  "lr": $LR,
  "proj_lr_mult": $PROJ_LR_MULT
}
EOF

echo "=== launch_8gpu_mm.sh ==="
echo "  OUT_DIR=$OUT_DIR"
echo "  mesh: NGPU=$NGPU FSDP=$FSDP PP=$PP TP=$TP CP=$CP EP=$EP V=$V"
echo "  recipe: STEPS=$STEPS LOCAL_BS=$LOCAL_BS GLOBAL_BS=$GLOBAL_BS SEQ_LEN=$SEQ_LEN"
echo "  trace: $TRACE_TIER (steps=$TRACE_STEPS)"
echo "========================="

export PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export HF_HOME="$HF_CACHE_DIR"
if [[ -n "$TRACE_ENV" ]]; then
    eval "export $TRACE_ENV"
fi

torchrun \
    --nproc_per_node="$NGPU" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter "$LOG_RANK" --role rank --tee 3 \
    -m phase5.train_mm \
    --mm.json "$JSON" \
    --mm.images "$IMAGES" \
    --mm.vision-model "$VISION" \
    --mm.tokenizer "$TOKENIZER" \
    --mm.cache-dir "$HF_CACHE_DIR" \
    --mm.proj-lr-mult "$PROJ_LR_MULT" \
    --mm.global-seq-len "$MM_GLOBAL_SEQ_LEN" \
    --module kimi_linear --config "$FLAVOR" \
    --hf_assets_path "$TORCHTITAN_DIR/assets/hf/Llama-3.1-8B" \
    --training.steps "$STEPS" \
    --training.local_batch_size "$LOCAL_BS" \
    --training.global_batch_size "$GLOBAL_BS" \
    --training.seq_len "$SEQ_LEN" \
    --optimizer.lr "$LR" \
    --lr_scheduler.warmup_steps "$WARMUP" \
    --lr_scheduler.total_steps "$STEPS" \
    --lr_scheduler.decay_ratio 0.0 \
    $PP_ARGS \
    --parallelism.data_parallel_shard_degree "$FSDP" \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree "$TP" \
    --parallelism.context_parallel_degree "$CP" \
    --parallelism.expert_parallel_degree "$EP" \
    $CKPT_ARGS \
    $DEBUG_ARGS \
    --metrics.save_tb_folder tb \
    --metrics.log_freq "$LOG_FREQ" \
    --dump_folder "$OUT_DIR" \
    $COMPILE_ARG \
    2>&1 | tee "$OUT_DIR/train.log"
