#!/usr/bin/env bash
# Phase 2 launch script: baseline + AttnRes Llama3-175M in a tmux session.
#
# Sequential, single-GPU. Each run writes its own TensorBoard + checkpoint
# dirs so results are never mixed up. One tmux session with separate
# windows per run so you can detach (Ctrl-b d), let it go overnight, and
# reattach (tmux attach -t attnres).
#
# Assumed layout (phase2_attnres_baseline_loss/ is a peer of torchtitan/):
#   <workspace>/
#   ├── phase2_attnres_baseline_loss/          <- this script lives here; outputs land under phase2_attnres_baseline_loss/runs/
#   └── torchtitan/      <- cloned fork, feat/block-attn-res branch
#
# Expected wall clock on RTX 5090, seq=2048, bs=16, 20k steps:
#   - Each run ~3-5 hours.
#   - Sequential total ~6-10 hours.
#
# Usage (from workspace root):
#   source /venv/main/bin/activate   # or: conda activate attnres
#   bash phase2_attnres_baseline_loss/launch.sh
#
# Attach / detach:
#   tmux attach -t attnres     # attach
#   Ctrl-b, d                  # detach
#   Ctrl-b, n / p / 0-9        # switch windows
#
# Tear down:
#   tmux kill-session -t attnres

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"
SESSION="${SESSION:-attnres}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/runs}"
STEPS="${STEPS:-20000}"
# Fraction of the planned training run to execute, in (0, 1].
# Multiplies STEPS (and thus tokens consumed, since C4 is streamed infinitely).
# Example: DATA_FRAC=0.125 on default STEPS=20000 -> 2500 steps per run,
# roughly ~1 hour total instead of ~8 hours, useful as a full-path dry run.
DATA_FRAC="${DATA_FRAC:-1.0}"
NGPU="${NGPU:-1}"
LOG_RANK="${LOG_RANK:-0}"
# Per-device micro-batch (config default 16 OOMs xent on 32GB RTX 5090 since
# logits materialize as [B*T, V=128256] in fp32). Use grad-accum to keep the
# effective batch size intact.
LOCAL_BS="${LOCAL_BS:-8}"
GLOBAL_BS="${GLOBAL_BS:-16}"
# Activation snippet for tmux windows. Defaults to /venv/main (the preinstalled
# venv on this box); override ACTIVATE="conda activate attnres" if you used the
# original conda path in setup_env.sh.
ACTIVATE="${ACTIVATE:-source /venv/main/bin/activate}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[launch] tmux not found. Install with: sudo apt install -y tmux"
    exit 1
fi

if [ ! -d "${TORCHTITAN_DIR}" ]; then
    echo "[launch] TORCHTITAN_DIR not found: ${TORCHTITAN_DIR}"
    exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[launch] tmux session '${SESSION}' already exists."
    echo "[launch] Kill it first with: tmux kill-session -t ${SESSION}"
    exit 1
fi

if [ ! -d "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" ]; then
    echo "[launch] Tokenizer not found at ${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B"
    echo "[launch] Run phase2_attnres_baseline_loss/setup_env.sh first."
    exit 1
fi

# Validate DATA_FRAC in (0, 1] and fold it into STEPS up-front so everything
# downstream (tmux command, GIT_SHA record, log messages) sees the same number.
EFFECTIVE_STEPS=$(/venv/main/bin/python - "$STEPS" "$DATA_FRAC" <<'PY'
import sys
steps = int(sys.argv[1])
frac = float(sys.argv[2])
if not (0.0 < frac <= 1.0):
    sys.exit(f"DATA_FRAC must be in (0, 1], got {frac}")
scaled = max(1, int(round(steps * frac)))
print(scaled)
PY
) || { echo "[launch] $EFFECTIVE_STEPS"; exit 1; }
echo "[launch] STEPS=${STEPS}  DATA_FRAC=${DATA_FRAC}  effective steps per run=${EFFECTIVE_STEPS}"
STEPS="${EFFECTIVE_STEPS}"

mkdir -p "${OUT_ROOT}/baseline" "${OUT_ROOT}/attn_res"

# Snapshot the git commit so we can correlate curves to code.
GIT_SHA=$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "${GIT_SHA}" > "${OUT_ROOT}/baseline/GIT_SHA"
echo "${GIT_SHA}" > "${OUT_ROOT}/attn_res/GIT_SHA"

# Shared launch recipe. Only --config and --dump_folder differ between runs.
# Each command cd's into TORCHTITAN_DIR first so torchrun imports find the
# package (we installed it editable, but cd'ing keeps relative paths sane).
make_cmd() {
    local config_name="$1"
    local dump_folder="$2"
    cat <<EOF
cd ${TORCHTITAN_DIR} && \\
PYTORCH_ALLOC_CONF="expandable_segments:True" \\
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d \\
    --rdzv_endpoint="localhost:0" \\
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \\
    -m torchtitan.train \\
    --module attention_residual \\
    --config ${config_name} \\
    --training.steps ${STEPS} \\
    --training.local_batch_size ${LOCAL_BS} \\
    --training.global_batch_size ${GLOBAL_BS} \\
    --dump_folder ${dump_folder} \\
    --metrics.save_tb_folder tb \\
    2>&1 | tee ${dump_folder}/train.log
EOF
}

echo "[launch] Creating tmux session: ${SESSION}"
tmux new-session -d -s "${SESSION}" -n baseline

# Window 1: baseline.
BASELINE_CMD="${ACTIVATE} && $(make_cmd llama3_175m_baseline ${OUT_ROOT}/baseline)"
tmux send-keys -t "${SESSION}:baseline" "${BASELINE_CMD}" C-m

# Window 2: AttnRes (waits for baseline DONE flag).
tmux new-window -t "${SESSION}" -n attn_res
ATTNRES_CMD=$(cat <<EOF
${ACTIVATE} && \
while ! [ -f ${OUT_ROOT}/baseline/DONE ]; do \
    echo "[attn_res] waiting for baseline to finish..."; \
    sleep 60; \
done && \
$(make_cmd llama3_175m_attn_res ${OUT_ROOT}/attn_res)
EOF
)
tmux send-keys -t "${SESSION}:attn_res" "${ATTNRES_CMD}" C-m

# Window 3: nvidia-smi monitor.
tmux new-window -t "${SESSION}" -n monitor
tmux send-keys -t "${SESSION}:monitor" "watch -n 2 nvidia-smi" C-m

# Window 4: guardian. Watches baseline/train.log for "Training completed"
# (see torchtitan/torchtitan/trainer.py:876) and touches DONE on match so
# the attn_res window unblocks.
tmux new-window -t "${SESSION}" -n guardian
GUARDIAN_CMD=$(cat <<EOF
while ! grep -q 'Training completed' ${OUT_ROOT}/baseline/train.log 2>/dev/null; do \
    sleep 30; \
done && \
touch ${OUT_ROOT}/baseline/DONE && \
echo '[guardian] baseline finished -- attn_res will now start'
EOF
)
tmux send-keys -t "${SESSION}:guardian" "${GUARDIAN_CMD}" C-m

echo "[launch] Session '${SESSION}' started"
echo "[launch]   windows: baseline, attn_res, monitor, guardian"
echo "[launch]   effective steps per run: ${STEPS} (DATA_FRAC=${DATA_FRAC})"
echo "[launch]   local_bs=${LOCAL_BS}  global_bs=${GLOBAL_BS}  grad_accum=$(( GLOBAL_BS / LOCAL_BS ))"
echo "[launch]   GIT_SHA: ${GIT_SHA}"
echo "[launch]   output:  ${OUT_ROOT}"
echo
echo "[launch] Attach:       tmux attach -t ${SESSION}"
echo "[launch] TensorBoard:  tensorboard --logdir ${OUT_ROOT} --port 6006 --bind_all"
echo "[launch] Compare:      python ${SCRIPT_DIR}/compare_losses.py \\"
echo "                          --baseline ${OUT_ROOT}/baseline/tb \\"
echo "                          --attn_res ${OUT_ROOT}/attn_res/tb \\"
echo "                          --out ${OUT_ROOT}/comparison.png"
