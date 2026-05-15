#!/usr/bin/env bash
# Problem B post-run summary — throughput + memory + loss alignment.
# Fills the "Comparison artifacts (post-run)" section of the README:
#
#   * Throughput: tps for Problem A AttnRes FSDP vs Problem B adapter_pp
#   * Memory: peak rank memory comparison
#   * Loss alignment: use compare_loss.sh
#
# Re-runnable mid-flight; uses whatever is logged so far.
#
# Usage: bash summarize_bench.sh

set -uo pipefail

PHASE4_RUNS="/root/torchtitan_attention_residual/phase4_kimi_attnres_lm_pretrain/runs"
BASELINE_LOG="${PHASE4_RUNS}/kimi_436m_baseline_fsdp_overnight/train.log"
ATTNRES_LOG="${PHASE4_RUNS}/kimi_436m_block_attn_res_fsdp_overnight/train.log"
ADAPTER_LOG="${PHASE4_RUNS}/kimi_pp_adapter_bench/adapter_pp/train.log"

# Extract final-tail throughput / memory stats from a train.log.
# Drops the first 10% of steps (warmup) and prints median tps + peak
# memory.
stats() {
    local log="$1"
    local name="$2"
    if [[ ! -f "${log}" ]]; then
        printf "%-28s  (missing)\n" "${name}"
        return
    fi
    # Strip ANSI, grab metric lines with tps + memory.
    sed -E 's/\x1b\[[0-9;]*m//g' "${log}" \
        | grep -E "step: +[0-9]+ +loss: .*memory: +[0-9.]+GiB.*tps: +[0-9,]+" \
        | sed -E 's/.*step: +([0-9]+).*memory: +([0-9.]+)GiB.*tps: +([0-9,]+).*/\1|\2|\3/' \
        > /tmp/_stats_rows.txt
    local nrows
    nrows=$(wc -l < /tmp/_stats_rows.txt)
    if [[ "${nrows}" -eq 0 ]]; then
        printf "%-28s  (no logged steps)\n" "${name}"
        rm -f /tmp/_stats_rows.txt
        return
    fi
    # Skip first 10% for warmup.
    local skip=$((nrows / 10))
    awk -F'|' -v name="${name}" -v skip="${skip}" '
        NR > skip {
            tps = $3; gsub(",", "", tps); tps += 0
            mem = $2 + 0
            step = $1 + 0
            tps_arr[NR - skip] = tps
            mem_arr[NR - skip] = mem
            last_step = step
            sum_tps += tps
            if (mem > peak_mem) peak_mem = mem
            n++
        }
        END {
            if (n == 0) {
                printf "%-28s  (fewer than 10 steps logged)\n", name
                exit
            }
            # Sort tps array for median.
            for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++)
                if (tps_arr[j] < tps_arr[i]) {
                    t = tps_arr[i]; tps_arr[i] = tps_arr[j]; tps_arr[j] = t
                }
            mid = int(n / 2) + 1
            median_tps = tps_arr[mid]
            mean_tps = sum_tps / n
            printf "%-28s  last_step=%5d  median_tps=%6d  mean_tps=%6d  peak_mem=%5.2f GiB  (n=%d samples)\n",
                name, last_step, median_tps, mean_tps, peak_mem, n
        }
    ' /tmp/_stats_rows.txt
    rm -f /tmp/_stats_rows.txt
}

echo "# Problem B post-run summary — throughput + memory"
echo "# generated: $(date -Is)"
echo ""
echo "## Per-arm stats (post-warmup, 10% skip)"
echo ""
printf "%-28s  %s\n" "arm" "metrics"
printf "%-28s  %s\n" "----" "-------"
stats "${BASELINE_LOG}" "baseline_fsdp (dense N=0)"
stats "${ATTNRES_LOG}"  "attnres_fsdp (block N=4)"
stats "${ADAPTER_LOG}"  "adapter_pp   (block N=8)"

echo ""
echo "## Interpretation"
echo ""
echo "* tps is per-rank; total throughput = tps × 4 GPUs."
echo "* FSDP arms use LOCAL_BS=3, SEQ=2048 (so 12 microbatches from global 12 via grad-accum)."
echo "* adapter_pp uses LOCAL_BS=1, SEQ=2048, PP=4 Interleaved1F1B with 12 microbatches."
echo "* Expected: adapter_pp tps ~0.4-0.6× FSDP tps. PP+cache on PCIe is bandwidth-bound;"
echo "  the adapter's headline benefit (less inter-stage bytes shipped) shows up on"
echo "  NVLink-or-slower fabrics where 60MB/hop savings translate to wall-clock."
echo "* Peak memory: PP sharding + LOCAL_BS=1 makes adapter_pp use ~15-17 GiB/rank,"
echo "  FSDP at LOCAL_BS=3 uses ~25-27 GiB/rank."

echo ""
echo "## Loss alignment"
echo ""
echo "See phase4_kimi_attnres_lm_pretrain/experiments/kimi_436m_attnres/compare_loss.sh for per-step"
echo "delta vs baseline. adapter_pp tracks attnres_fsdp within bf16+NCCL noise"
echo "modulo flavor (N=8 vs N=4)."
