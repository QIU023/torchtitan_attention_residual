#!/usr/bin/env bash
# Phase 2 num_blocks ablation: run several AttnRes variants sequentially
# in one tmux session. Each variant is a full-scale 20k step run; results
# land under ``runs/ablation/<variant>/`` for downstream compare.
#
# Usage (from workspace root):
#   bash phase2_attnres_baseline_loss/launch_ablation.sh
#
# Override the variant list:
#   VARIANTS="llama3_175m_attn_res_n3 llama3_175m_attn_res_n4" \
#       bash phase2_attnres_baseline_loss/launch_ablation.sh
#
# Override steps (for a shorter smoke):
#   STEPS=2000 bash phase2_attnres_baseline_loss/launch_ablation.sh
#
# Reattach: tmux attach -t ablation
# Detach:   Ctrl-b d
# Kill:     tmux kill-session -t ablation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"
SESSION="${SESSION:-ablation}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/runs/ablation}"
STEPS="${STEPS:-20000}"
LOCAL_BS="${LOCAL_BS:-8}"
GLOBAL_BS="${GLOBAL_BS:-16}"
NGPU="${NGPU:-1}"
ACTIVATE="${ACTIVATE:-source /venv/main/bin/activate}"

# Two-ended ablation by default (tightest RFC value per hour). The user
# already has N=6 from the primary run in ../runs/attn_res/.
VARIANTS="${VARIANTS:-llama3_175m_attn_res_n3 llama3_175m_attn_res_n12}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ablation] tmux not found. Install with: sudo apt install -y tmux"
    exit 1
fi

if [ ! -d "${TORCHTITAN_DIR}" ]; then
    echo "[ablation] TORCHTITAN_DIR not found: ${TORCHTITAN_DIR}"
    exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[ablation] tmux session '${SESSION}' already exists."
    echo "[ablation] Kill it first: tmux kill-session -t ${SESSION}"
    exit 1
fi

GIT_SHA=$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
mkdir -p "${OUT_ROOT}"

# Build an unconditional chain: each variant runs whether or not the
# previous succeeded. Each segment captures its torchrun exit code into
# a per-variant STATUS file so you can tell after the fact which ones
# finished and which failed. Chaining is via `;` (not `&&`) precisely so
# a mid-run crash of variant N does NOT prevent variant N+1 from starting.
chain_cmd=""
for variant in ${VARIANTS}; do
    run_dir="${OUT_ROOT}/${variant}"
    mkdir -p "${run_dir}"
    echo "${GIT_SHA}" > "${run_dir}/GIT_SHA"

    train_cmd=$(cat <<EOF
echo "[ablation] === starting ${variant} (steps=${STEPS}) ==="; \\
cd ${TORCHTITAN_DIR}; \\
PYTORCH_ALLOC_CONF="expandable_segments:True" \\
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d \\
    --rdzv_endpoint="localhost:0" \\
    --local-ranks-filter 0 --role rank --tee 3 \\
    -m torchtitan.train \\
    --module attn_res \\
    --config ${variant} \\
    --training.steps ${STEPS} \\
    --training.local_batch_size ${LOCAL_BS} \\
    --training.global_batch_size ${GLOBAL_BS} \\
    --dump_folder ${run_dir} \\
    --metrics.save_tb_folder tb \\
    > >(tee ${run_dir}/train.log) 2>&1; \\
RC=\$?; \\
echo "[ablation] === ${variant} exited rc=\$RC ==="; \\
echo "\$RC" > ${run_dir}/STATUS; \\
if [ \$RC -eq 0 ]; then touch ${run_dir}/DONE; fi
EOF
)
    if [ -z "${chain_cmd}" ]; then
        chain_cmd="${train_cmd}"
    else
        chain_cmd="${chain_cmd} ; ${train_cmd}"
    fi
done
# Append a final marker so you can grep for 'ablation: all done' in the
# session output to know the whole sweep finished (even if individual
# variants failed).
chain_cmd="${chain_cmd} ; echo '[ablation: all variants attempted; see STATUS files]' ; touch ${OUT_ROOT}/SWEEP_DONE"

tmux new-session -d -s "${SESSION}" -n sweep
# Use bash -c so the tmux window runs our chain inside a single shell,
# and set +e so one variant's crash does not exit the shell before the
# next variant's segment runs.
tmux send-keys -t "${SESSION}:sweep" "${ACTIVATE} && bash -c 'set +e; ${chain_cmd}'" C-m

tmux new-window -t "${SESSION}" -n monitor
tmux send-keys -t "${SESSION}:monitor" "watch -n 2 nvidia-smi" C-m

echo "[ablation] Session '${SESSION}' started"
echo "[ablation]   variants:   ${VARIANTS}"
echo "[ablation]   steps each: ${STEPS}"
echo "[ablation]   out_root:   ${OUT_ROOT}"
echo "[ablation]   git_sha:    ${GIT_SHA}"
echo
echo "[ablation] Attach:  tmux attach -t ${SESSION}"
echo "[ablation] Watch done flags:"
for variant in ${VARIANTS}; do
    echo "  ls ${OUT_ROOT}/${variant}/DONE 2>/dev/null"
done
