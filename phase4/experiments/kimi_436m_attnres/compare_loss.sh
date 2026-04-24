#!/usr/bin/env bash
# Compare baseline vs AttnRes Kimi Linear 436M FSDP runs.
# Re-runnable: safe to invoke while AttnRes is mid-flight; it
# reports whatever has been logged so far.
#
# Emits to stdout:
#   1. CSV per matched step: step, baseline_loss, attnres_loss, delta_loss,
#      baseline_grad, attnres_grad, baseline_tps, attnres_tps
#   2. ASCII milestone summary at the tail
#
# Usage: bash compare_loss.sh [step_stride]
#   step_stride default 1000 (report every 1000 steps)

set -uo pipefail

STRIDE=${1:-1000}

PHASE4_RUNS="/root/torchtitan_attention_residual/phase4/runs"
BASELINE_LOG="${PHASE4_RUNS}/kimi_436m_baseline_fsdp_overnight/train.log"
ATTNRES_LOG="${PHASE4_RUNS}/kimi_436m_block_attn_res_fsdp_overnight/train.log"

for f in "${BASELINE_LOG}" "${ATTNRES_LOG}"; do
    [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

TMP_BASELINE=$(mktemp)
TMP_ATTNRES=$(mktemp)
trap 'rm -f "${TMP_BASELINE}" "${TMP_ATTNRES}"' EXIT

# Extract (step, loss, grad, tps) for steps matching the stride.
# Strip ANSI color codes, pull fields with sed, filter in awk.
extract() {
    local log="$1"
    local out="$2"
    sed -E 's/\x1b\[[0-9;]*m//g' "${log}" \
        | grep -E "step: +[0-9]+ +loss: +[0-9.]+.*grad_norm: +[0-9.]+.*tps: +[0-9,]+" \
        | sed -E 's/.*step: +([0-9]+) +loss: +([0-9.]+).*grad_norm: +([0-9.]+).*tps: +([0-9,]+).*$/\1|\2|\3|\4/' \
        | awk -F'|' -v stride="${STRIDE}" '
            {
                step = $1 + 0
                if (step % stride == 0) {
                    tps = $4; gsub(",", "", tps); tps += 0
                    printf "%d %s %s %d\n", step, $2, $3, tps
                }
            }' \
        | sort -k1,1n -u > "${out}"
}

extract "${BASELINE_LOG}" "${TMP_BASELINE}"
extract "${ATTNRES_LOG}" "${TMP_ATTNRES}"

echo "# baseline vs attnres — kimi linear 436m fsdp"
echo "# generated: $(date -Is)"
echo "# baseline log: ${BASELINE_LOG}"
echo "# attnres  log: ${ATTNRES_LOG}"
echo ""
echo "step,baseline_loss,attnres_loss,delta_loss,baseline_grad,attnres_grad,baseline_tps,attnres_tps"

# Merge on step key via awk, two-pass.
awk '
    NR == FNR {
        base_loss[$1] = $2; base_grad[$1] = $3; base_tps[$1] = $4
        next
    }
    $1 in base_loss {
        delta = $2 - base_loss[$1]
        printf "%d,%s,%s,%+.4f,%s,%s,%d,%d\n", $1, base_loss[$1], $2, delta, base_grad[$1], $3, base_tps[$1], $4
    }
' "${TMP_BASELINE}" "${TMP_ATTNRES}" > /tmp/_compare_out.csv

cat /tmp/_compare_out.csv

echo ""
echo "# --- summary ---"
echo "# negative delta = attnres below baseline (better)"

if [[ -s /tmp/_compare_out.csv ]]; then
    tail -1 /tmp/_compare_out.csv | awk -F, '{
        printf "# latest matched step: %d | baseline=%.4f attnres=%.4f delta=%+.4f\n", $1, $2, $3, $4
    }'
    tail -5 /tmp/_compare_out.csv | awk -F, '{ sum += $4; n++ } END { if (n>0) printf "# mean delta over last %d matched points: %+.4f\n", n, sum/n }'
else
    echo "# no matched steps yet"
fi

rm -f /tmp/_compare_out.csv
