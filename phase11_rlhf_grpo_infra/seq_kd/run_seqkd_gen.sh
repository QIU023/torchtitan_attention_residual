#!/usr/bin/env bash
# seq-KD teacher generation — 8 single-GPU vLLM replicas (data parallel, NO TP).
#
# 5090 has no NVLink → TP across PCIe is bandwidth-bound. Instead each GPU runs a
# full AWQ-4bit Qwen3-VL-30B-A3B copy (TP=1) and chews its 1/8 shard of the data.
# Zero inter-GPU comms.
#
# Usage:
#   bash run_seqkd_gen.sh smoke        # 5 conv on GPU0, sanity check
#   bash run_seqkd_gen.sh full         # all 665k across 8 GPUs
set -uo pipefail
cd /workspace/torchtitan_attention_residual
export HF_HOME=/workspace/.hf_home
ulimit -c 0

MODE="${1:-smoke}"
SKD="phase11_rlhf_grpo_infra/seq_kd"
INPUT_JSON="/workspace/.hf_home/LLaVA-Instruct/llava_v1_5_mix665k.json"
IMAGE_ROOT="/workspace/.hf_home/LLaVA-Instruct/images"
OUT_DIR="${SKD}/out_${MODE}"
LOG_DIR="${SKD}/logs/${MODE}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# resolve teacher snapshot path
MODEL_DIR=$(ls -d /workspace/.hf_home/hub/models--QuantTrio--Qwen3-VL-30B-A3B-Instruct-AWQ/snapshots/*/ 2>/dev/null | head -1)
if [[ -z "$MODEL_DIR" || ! -f "$MODEL_DIR/config.json" ]]; then
    echo "FATAL: teacher snapshot not found / incomplete: $MODEL_DIR"; exit 1
fi
echo "[seqkd] teacher = $MODEL_DIR"
echo "[seqkd] mode = $MODE"

MAXNEW="${MAXNEW:-512}"
TEMP="${TEMP:-0.0}"

if [[ "$MODE" == "smoke" ]]; then
    echo "[seqkd] SMOKE: 5 conversations on GPU0"
    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 "$SKD/gen_worker.py" \
        --model "$MODEL_DIR" --input-json "$INPUT_JSON" --image-root "$IMAGE_ROOT" \
        --out-jsonl "$OUT_DIR/shard0.jsonl" \
        --shard-id 0 --num-shards 8 --limit 5 \
        --max-new-tokens "$MAXNEW" --temperature "$TEMP" \
        2>&1 | tee "$LOG_DIR/shard0.log"
    echo "=== SMOKE OUTPUT ==="
    /usr/bin/python3 - "$OUT_DIR/shard0.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    r=json.loads(line)
    print("id:",r.get("id"),"image:",r.get("image"))
    for m in r["conversations"]:
        v=m["value"][:200].replace("\n"," ")
        print(f"  [{m['from']}] {v}")
    print("-"*60)
PY
    exit 0
fi

# ---- FULL: 8 replicas ----
# disk watchdog
(
    while true; do
        sleep 120
        F=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
        if (( F < 10 )); then
            echo "[watchdog] PANIC disk ${F}G; killing gen"
            pkill -9 -f gen_worker.py 2>/dev/null
            touch "$OUT_DIR/DISK_PANIC"; exit 1
        fi
    done
) &
WD=$!
trap 'kill -9 ${WD} 2>/dev/null' EXIT

echo "[seqkd] launching 8 replicas…"
PIDS=()
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i /usr/bin/python3 "$SKD/gen_worker.py" \
        --model "$MODEL_DIR" --input-json "$INPUT_JSON" --image-root "$IMAGE_ROOT" \
        --out-jsonl "$OUT_DIR/shard${i}.jsonl" \
        --shard-id "$i" --num-shards 8 \
        --max-new-tokens "$MAXNEW" --temperature "$TEMP" \
        > "$LOG_DIR/shard${i}.log" 2>&1 &
    PIDS+=($!)
    sleep 8   # stagger model loads to avoid disk/CPU thrash
done
echo "[seqkd] PIDs: ${PIDS[*]}"

fail=0
for p in "${PIDS[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[seqkd] all replicas exited (failures=$fail)"

# ---- merge shards → distilled json ----
echo "[seqkd] merging shards → distilled json"
/usr/bin/python3 - "$OUT_DIR" "$SKD/distilled_mix665k_${MODE}.json" <<'PY'
import json,sys,glob,os
out_dir, dest = sys.argv[1], sys.argv[2]
recs=[]
for f in sorted(glob.glob(os.path.join(out_dir,"shard*.jsonl"))):
    for line in open(f):
        try: recs.append(json.loads(line))
        except Exception: pass
json.dump(recs, open(dest,"w"), ensure_ascii=False)
print(f"merged {len(recs)} conversations -> {dest}")
PY
echo "[seqkd] FULL gen done."
