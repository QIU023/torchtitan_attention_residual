#!/usr/bin/env bash
# STAGE 1 — seq-KD teacher generation on 2xH200 (DP=2, NOT TP).
# Qwen3-VL-30B-A3B-AWQ (~17GB) fits on one 144GB H200, so we run 2 TP=1
# replicas (data-parallel: each chews half the shards, zero inter-GPU comms).
# Throughput-optimal for offline batch generation.
#
# Env:
#   SUBSET    total conversations to distill across all shards (0 = full 665k)
#   NUM_SHARDS number of GPU replicas (default 2)
#   MAXNEW TEMP GPUMEM MAXLEN
# Runs under the vLLM venv (separate from the training conda env).
set -uo pipefail
ulimit -c 0
source /home/seqkd_overnight/lib.sh

SKD="${REPO}/phase11_rlhf_grpo_infra/seq_kd"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0

INPUT_JSON=/home/.hf_home/LLaVA-Instruct/llava_v1_5_mix665k.json
IMAGE_ROOT=/home/.hf_home/LLaVA-Instruct/images
MODE="${MODE:-full}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SUBSET="${SUBSET:-0}"
OUT_DIR="${SKD}/out_${MODE}"
LOG_DIR="${ROOT}/logs/gen_${MODE}"
RAW_JSON="${SKD}/distilled_mix665k_${MODE}_raw.json"
DEST_JSON="${SKD}/distilled_mix665k_${MODE}.json"
mkdir -p "$OUT_DIR" "$LOG_DIR"

MODEL_DIR=$(ls -d /home/.hf_home/hub/models--QuantTrio--Qwen3-VL-30B-A3B-Instruct-AWQ/snapshots/*/ 2>/dev/null | head -1)
if [[ -z "$MODEL_DIR" || ! -f "$MODEL_DIR/config.json" ]]; then
    log "FATAL: teacher snapshot not found/incomplete: $MODEL_DIR"; exit 1
fi
log "[gen] teacher=$MODEL_DIR mode=$MODE shards=$NUM_SHARDS subset=$SUBSET"

MAXNEW="${MAXNEW:-512}"; TEMP="${TEMP:-0.0}"; GPUMEM="${GPUMEM:-0.90}"; MAXLEN="${MAXLEN:-8192}"
CHUNK="${CHUNK:-2000}"; MAXSEQS="${MAXSEQS:-256}"

# per-shard limit: each shard sees floor(SUBSET/NUM_SHARDS) of its slice
LIMIT_ARG=""
if (( SUBSET > 0 )); then
    PER=$(( (SUBSET + NUM_SHARDS - 1) / NUM_SHARDS ))
    LIMIT_ARG="--limit ${PER}"
    log "[gen] per-shard limit=${PER}"
fi

log "[gen] launching ${NUM_SHARDS} replicas..."
PIDS=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i "$VPY" "$SKD/gen_worker.py" \
        --model "$MODEL_DIR" --input-json "$INPUT_JSON" --image-root "$IMAGE_ROOT" \
        --out-jsonl "$OUT_DIR/shard${i}.jsonl" \
        --shard-id "$i" --num-shards "$NUM_SHARDS" ${LIMIT_ARG} \
        --max-new-tokens "$MAXNEW" --temperature "$TEMP" \
        --gpu-mem-util "$GPUMEM" --max-model-len "$MAXLEN" \
        --chunk "$CHUNK" --max-num-seqs "$MAXSEQS" \
        > "$LOG_DIR/shard${i}.log" 2>&1 &
    PIDS+=($!)
    sleep 12   # stagger model loads
done
log "[gen] PIDs: ${PIDS[*]}"
fail=0
for p in "${PIDS[@]}"; do wait "$p" || fail=$((fail+1)); done
log "[gen] replicas exited (failures=$fail)"

# merge shards -> raw json
log "[gen] merging shards -> $RAW_JSON"
"$VPY" - "$OUT_DIR" "$RAW_JSON" <<'PY'
import json,sys,glob,os
out_dir, dest = sys.argv[1], sys.argv[2]
recs=[]
for f in sorted(glob.glob(os.path.join(out_dir,"shard*.jsonl"))):
    for line in open(f):
        try: recs.append(json.loads(line))
        except Exception: pass
json.dump(recs, open(dest,"w"), ensure_ascii=False)
print(f"merged {len(recs)} -> {dest}")
PY

# filter to rows whose image exists (so student SFT never hits a missing file)
log "[gen] filtering to existing-image rows -> $DEST_JSON"
"$VPY" /home/seqkd_overnight/filter_existing_images.py "$RAW_JSON" "$DEST_JSON" "$IMAGE_ROOT"
N=$("$VPY" -c "import json;print(len(json.load(open('$DEST_JSON'))))")
log "[gen] FINAL distilled rows=$N failures=$fail"
exit $fail
