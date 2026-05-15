#!/usr/bin/env bash
# Compare Kimi Linear 436M runs across Problem A (FSDP baseline + AttnRes
# N=4) and Problem B (PP+adapter AttnRes N=8). Re-runnable: safe to
# invoke while any run is mid-flight; reports whatever has been logged
# so far.
#
# Note: Problem B uses N=8 (block_attn_res, no _n4 suffix) while Problem A
# AttnRes uses N=4 — so the B-vs-A comparison isn't a pure "FSDP vs PP"
# a/b on the same model. It's the best available reference and still
# reveals whether the cache-adapter stack is fundamentally training
# correctly (loss trajectory sanity).
#
# Emits to stdout:
#   1. CSV per matched step with cols for whichever logs are available
#   2. ASCII milestone summary at the tail
#
# Usage: bash compare_loss.sh [step_stride]
#   step_stride default 1000 (report every 1000 steps)

set -uo pipefail

STRIDE=${1:-1000}

PHASE4_RUNS="/root/torchtitan_attention_residual/phase4_kimi_attnres_lm_pretrain/runs"
BASELINE_LOG="${PHASE4_RUNS}/kimi_436m_baseline_fsdp_overnight/train.log"
ATTNRES_LOG="${PHASE4_RUNS}/kimi_436m_block_attn_res_fsdp_overnight/train.log"
ADAPTER_LOG="${PHASE4_RUNS}/kimi_pp_adapter_bench/adapter_pp/train.log"

[[ -f "${BASELINE_LOG}" ]] || { echo "missing: ${BASELINE_LOG}" >&2; exit 1; }
[[ -f "${ATTNRES_LOG}"  ]] || { echo "missing: ${ATTNRES_LOG}"  >&2; exit 1; }
HAS_ADAPTER=0
[[ -f "${ADAPTER_LOG}"  ]] && HAS_ADAPTER=1

TMP_BASELINE=$(mktemp)
TMP_ATTNRES=$(mktemp)
TMP_ADAPTER=$(mktemp)
trap 'rm -f "${TMP_BASELINE}" "${TMP_ATTNRES}" "${TMP_ADAPTER}" /tmp/_compare_out.csv' EXIT

# Extract (step, loss, grad, tps) for steps matching the stride.
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
if [[ "${HAS_ADAPTER}" = "1" ]]; then
    extract "${ADAPTER_LOG}" "${TMP_ADAPTER}"
fi

echo "# kimi linear 436m — three-way run comparison"
echo "# generated: $(date -Is)"
echo "# A baseline FSDP : ${BASELINE_LOG}"
echo "# A attnres  FSDP : ${ATTNRES_LOG}"
if [[ "${HAS_ADAPTER}" = "1" ]]; then
    echo "# B adapter_pp N=8: ${ADAPTER_LOG} (present)"
else
    echo "# B adapter_pp N=8: ${ADAPTER_LOG} (missing — run not started)"
fi
echo ""

if [[ "${HAS_ADAPTER}" = "1" ]]; then
    echo "step,baseline_loss,attnres_loss,adapter_loss,delta_attnres,delta_adapter,baseline_tps,attnres_tps,adapter_tps"
    # three-way merge driven by the ADAPTER file's steps (it's the newest; pre-sorted numerically).
    awk -v fb="${TMP_BASELINE}" -v fa="${TMP_ATTNRES}" '
        BEGIN {
            while ((getline line < fb) > 0) {
                split(line, a, " ")
                base_loss[a[1]] = a[2]; base_tps[a[1]] = a[4]
            }
            close(fb)
            while ((getline line < fa) > 0) {
                split(line, a, " ")
                attn_loss[a[1]] = a[2]; attn_tps[a[1]] = a[4]
            }
            close(fa)
        }
        {
            s = $1
            if ((s in base_loss) && (s in attn_loss)) {
                da = attn_loss[s] - base_loss[s]
                dp = $2 - base_loss[s]
                printf "%d,%s,%s,%s,%+.4f,%+.4f,%d,%d,%d\n",
                    s, base_loss[s], attn_loss[s], $2, da, dp, base_tps[s], attn_tps[s], $4
            }
        }
    ' "${TMP_ADAPTER}" > /tmp/_compare_out.csv
else
    echo "step,baseline_loss,attnres_loss,delta_loss,baseline_grad,attnres_grad,baseline_tps,attnres_tps"
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
fi

cat /tmp/_compare_out.csv

echo ""
echo "# --- summary ---"
echo "# delta_attnres = attnres - baseline (negative = attnres better)"
if [[ "${HAS_ADAPTER}" = "1" ]]; then
    echo "# delta_adapter = adapter_pp - baseline (flavor differs: N=8 vs N=4; loose comparison)"
fi

if [[ -s /tmp/_compare_out.csv ]]; then
    LAST=$(tail -1 /tmp/_compare_out.csv)
    if [[ "${HAS_ADAPTER}" = "1" ]]; then
        echo "${LAST}" | awk -F, '{
            printf "# latest matched step: %d | baseline=%.4f attnres=%.4f adapter=%.4f | delta_attn=%+.4f delta_adp=%+.4f\n",
                $1, $2, $3, $4, $5, $6
        }'
    else
        echo "${LAST}" | awk -F, '{
            printf "# latest matched step: %d | baseline=%.4f attnres=%.4f delta=%+.4f\n", $1, $2, $3, $4
        }'
    fi
else
    echo "# no matched steps yet across runs"
fi
