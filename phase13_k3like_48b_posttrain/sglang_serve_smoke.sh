#!/bin/bash
# Smoke: does the AttnRes/Kimi-Linear sglang overlay serve our model on
# the 5090? Launches the isolated sglang server (torch 2.11 venv) on the
# 194m official-format checkpoint (1.45 GB, fits one 32 GB card), waits
# for readiness, sends one /generate, prints the completion. The 48B does
# NOT fit a single 5090 for inference -- this proves the overlay/model
# registration + KDA MambaRadixCache path, which is arch-identical at 48B.
set -eo pipefail
VENV=/workspace/sgl_venv
MODEL=/workspace/fake_hf/kimi_linear_194m
PORT=${PORT:-30000}
LOG=/workspace/sglang_server.log

pkill -f "sglang.launch_server" 2>/dev/null || true
echo "[serve] launching sglang server on 127.0.0.1:${PORT} ..."
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" \
  --mem-fraction-static 0.6 --max-running-requests 4 \
  --attention-backend triton --disable-cuda-graph \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "[serve] pid ${SERVER_PID}, waiting for readiness (log: ${LOG})"

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/get_model_info" >/dev/null 2>&1; then
    echo "[serve] READY after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] SERVER DIED -- last log:"; tail -30 "$LOG"; exit 1
  fi
  sleep 1
done

echo "[serve] /generate ..."
curl -sf "http://127.0.0.1:${PORT}/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('[serve] OUTPUT:', repr(d.get('text','')[:120]))"

echo "[serve] SGLANG_SERVE_SMOKE_PASS"
kill "$SERVER_PID" 2>/dev/null || true
