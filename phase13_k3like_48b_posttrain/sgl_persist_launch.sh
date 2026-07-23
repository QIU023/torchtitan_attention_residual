#!/bin/bash
VENV=/workspace/sgl_venv; MODEL=/workspace/fake_hf/kimi_linear_194m; PORT=30000
pkill -f "sglang.launch_server" 2>/dev/null; sleep 2
exec env CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code --host 127.0.0.1 --port "$PORT" \
  --mem-fraction-static 0.6 --max-running-requests 8 \
  --attention-backend triton --disable-cuda-graph
