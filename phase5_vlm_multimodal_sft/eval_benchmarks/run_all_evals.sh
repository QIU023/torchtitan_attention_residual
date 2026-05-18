#!/usr/bin/env bash
# Run all VLM downstream eval benchmarks on a stage-2 SFT checkpoint.
#
# Invokes each score_<bench>.py under torchrun (8 GPUs, 1D FSDP), waits
# for the per-rank prediction JSONLs to land, then aggregates result.json
# files into a single REPORT.md.
#
# Usage:
#   STAGE2_CKPT=/abs/path/to/checkpoint/step-3800 ./run_all_evals.sh
# Optional knobs:
#   RUN_DIR=...           where per-bench outputs go (default: ./runs/<timestamp>)
#   BENCHES="pope gqa ..."  override which benches to run
#   POPE_LIMIT=0          cap POPE records (smoke). Default 0 = all 9K.
#   GQA_LIMIT=0           cap GQA records.
#   MMB_LIMIT=0           cap MMBench records.
#   SQA_LIMIT=0           cap ScienceQA records.
#   MMMU_LIMIT=0          cap MMMU records.
set -euo pipefail
ulimit -c 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}"
PHASE5_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PHASE5_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

STAGE2_CKPT="${STAGE2_CKPT:?STAGE2_CKPT required (full path to a step-N dir)}"
[[ -d "${STAGE2_CKPT}" ]] || { echo "ERROR: ckpt missing: ${STAGE2_CKPT}"; exit 1; }

TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-${EVAL_DIR}/runs/${TS}_$(basename ${STAGE2_CKPT})}"
mkdir -p "${RUN_DIR}"
echo "[$(date)] run_all_evals: ckpt=${STAGE2_CKPT} run_dir=${RUN_DIR}"

STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_447m_aligned_block_attn_res_n4_fp8}"
INSTRUCT_DIR="${INSTRUCT_DIR:-/workspace/.hf_home/LLaVA-Instruct}"
JSON="${JSON:-${INSTRUCT_DIR}/llava_v1_5_mix665k.json}"
IMAGES="${IMAGES:-${INSTRUCT_DIR}/images}"
VISION="${VISION:-google/siglip-base-patch16-224}"
TOKENIZER="${TOKENIZER:-NousResearch/Meta-Llama-3.1-8B}"
CACHE_DIR="${CACHE_DIR:-/workspace/.hf_home}"

NGPU="${NGPU:-8}"
# For eval we need local_batch_size * NGPU == global_batch_size and the
# trainer needs a non-zero steps to not exit. Both are stub values.
LOCAL_BS=1
GLOBAL_BS="${NGPU}"
SEQ_LEN="${SEQ_LEN:-580}"     # placeholder; we don't actually train.
TARGET_STEPS=1                # placeholder; load_ckpt_only() handles loading.

BENCHES="${BENCHES:-mmmu scienceqa mmbench pope gqa}"

# Per-bench config: (module, out_subdir, extra_limit_env_var, default_max_new_tokens,
# wall-clock timeout in seconds for the entire torchrun invocation — caps any
# rank stall so the orchestrator keeps moving and the partial-aggregator can
# still score the records that finished.)
run_bench () {
    local bench="$1"
    local module
    local out_sub
    local limit_var
    local bench_timeout
    case "${bench}" in
        pope)      module="score_pope";     out_sub="pope";          limit_var="POPE_LIMIT"; bench_timeout="${POPE_TIMEOUT:-3600}"  ;;
        gqa)       module="score_gqa";      out_sub="gqa";           limit_var="GQA_LIMIT";  bench_timeout="${GQA_TIMEOUT:-3600}"   ;;
        mmbench)   module="score_mmbench";  out_sub="mmbench_en_dev";limit_var="MMB_LIMIT";  bench_timeout="${MMB_TIMEOUT:-1800}"   ;;
        scienceqa) module="score_scienceqa";out_sub="scienceqa_img"; limit_var="SQA_LIMIT";  bench_timeout="${SQA_TIMEOUT:-900}"    ;;
        mmmu)      module="score_mmmu";     out_sub="mmmu_val";      limit_var="MMMU_LIMIT"; bench_timeout="${MMMU_TIMEOUT:-600}"   ;;
        *) echo "WARN: unknown bench '${bench}', skipping"; return 0 ;;
    esac
    local out_dir="${RUN_DIR}/${out_sub}"
    mkdir -p "${out_dir}"
    local limit="${!limit_var:-0}"

    echo "================================================================"
    echo "[$(date)] bench=${bench}  out=${out_dir}  limit=${limit}"
    echo "================================================================"

    # EACH bench launches its own torchrun. We deliberately re-pay
    # process startup so a crash in one bench doesn't break the others.
    set +e
    # tee=3 means stderr+stdout shown for filtered ranks; we leave the
    # filter at rank 0 only to keep logs readable, but each rank's
    # error_file landed under ${out_dir}/_titan_dump/ in torchrun's
    # default crash dump path.
    PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200 \
    NCCL_TIMEOUT=1200 \
    HF_HOME="${HF_HOME:-/workspace/.hf_home}" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    timeout --foreground --kill-after=60 "${bench_timeout}" \
    /usr/local/bin/torchrun \
        --nproc_per_node="${NGPU}" \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        --local-ranks-filter 0 --role rank --tee 3 \
        --redirects 3 --log-dir "${out_dir}/_torchrun_logs" \
        -m phase5_vlm_multimodal_sft.eval_benchmarks.${module} \
        --eval.output-dir "${out_dir}" \
        --eval.limit "${limit}" \
        --mm.json "${JSON}" \
        --mm.images "${IMAGES}" \
        --mm.vision-model "${VISION}" \
        --mm.tokenizer "${TOKENIZER}" \
        --mm.cache-dir "${CACHE_DIR}" \
        --mm.proj-lr-mult 1.0 \
        --mm.global-seq-len "${SEQ_LEN}" \
        --mm.layout sft \
        --mm.val-samples 0 \
        --mm.val-stratified-per-source 0 \
        --mm.val-freq 0 \
        --mm.val-batches 0 \
        --module kimi_linear --config "${STUDENT_CONFIG}" \
        --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
        --training.steps "${TARGET_STEPS}" \
        --training.local_batch_size "${LOCAL_BS}" \
        --training.global_batch_size "${GLOBAL_BS}" \
        --training.seq_len "${SEQ_LEN}" \
        --training.max_norm 1.0 \
        --parallelism.data_parallel_shard_degree "${NGPU}" \
        --parallelism.fsdp_reshard_after_forward never \
        --optimizer.lr 1e-5 \
        --lr_scheduler.warmup_steps 1 \
        --lr_scheduler.decay_ratio 0.2 \
        --lr_scheduler.min_lr_factor 0.1 \
        --checkpoint.enable \
        --checkpoint.interval 999999 \
        --checkpoint.keep_latest_k 2 \
        --checkpoint.initial_load_path "${STAGE2_CKPT}" \
        --checkpoint.initial_load_model_only \
        --metrics.log_freq 999999 \
        --metrics.save_tb_folder tb \
        --dump_folder "${out_dir}/_titan_dump" 2>&1 | tee "${out_dir}/run.log"
    local rc="${PIPESTATUS[0]}"
    set -e
    if [[ "${rc}" != "0" ]]; then
        echo "[$(date)] bench=${bench} torchrun FAILED rc=${rc} (likely one rank crashed); attempting partial-aggregate"
    else
        echo "[$(date)] bench=${bench} torchrun done"
    fi

    # Always post-process: aggregate whatever preds_rank*.jsonl exist and
    # score them. Survives mid-run rank crashes that took down torchrun.
    PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    /usr/bin/python3 -m phase5_vlm_multimodal_sft.eval_benchmarks.postprocess \
        --bench "${bench}" \
        --output-dir "${out_dir}" \
        --limit "${limit}" 2>&1 | tee -a "${out_dir}/run.log"
}

for bench in ${BENCHES}; do
    run_bench "${bench}" || true
done

# ---- Aggregate REPORT.md ----
echo "================================================================"
echo "[$(date)] aggregating report → ${RUN_DIR}/REPORT.md"
echo "================================================================"

PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
/usr/bin/python3 -m phase5_vlm_multimodal_sft.eval_benchmarks.aggregate_report \
    --run-dir "${RUN_DIR}" \
    --ckpt "${STAGE2_CKPT}" \
    --out "${RUN_DIR}/REPORT.md"

# Symlink latest for convenience
ln -sfn "${RUN_DIR}" "${EVAL_DIR}/runs/LATEST"
ln -sfn "${RUN_DIR}/REPORT.md" "${EVAL_DIR}/REPORT.md"

echo "[$(date)] DONE → ${RUN_DIR}/REPORT.md"
