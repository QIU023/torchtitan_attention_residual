#!/usr/bin/env bash
# Kimi Linear 48B downscale feasibility sweep on 8×32 GiB.
#
# Starts at paper dim=2304 with reduced num_experts; falls back to
# smaller dim variants if everything OOMs. Each combo runs 3 steps
# (or until OOM). Goal: find largest single-node-feasible (dim,
# num_experts) combo with paper-aligned L=27 / N=9 layout.
#
# Paper hard-coded: n_layers=27, num_blocks=9 (Block AttnRes, 3
# t-blocks/AttnRes-block sweet spot), seq_len=4096.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase4_kimi_attnres_lm_pretrain/run_kimi48b_downscale_sweep.log"
> "$LOG"
exec >>"$LOG" 2>&1

SEQ_LEN="${SEQ_LEN:-4096}"

run_one() {
    local flavor="$1"
    local out="/tmp/${flavor}_seq${SEQ_LEN}"
    rm -rf "$out"
    echo ""
    echo "==============================================================="
    echo "[$(date)] $flavor seq=${SEQ_LEN} START"
    echo "==============================================================="
    (cd torchtitan && torchrun \
        --nproc_per_node=8 \
        --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
        --local-ranks-filter 7 --role rank --tee 3 \
        -m torchtitan.train \
        --module attention_residual --config "$flavor" \
        --training.steps 3 \
        --training.local_batch_size 1 \
        --training.global_batch_size 8 \
        --training.seq_len "${SEQ_LEN}" \
        --parallelism.data_parallel_shard_degree 8 \
        --parallelism.expert_parallel_degree 8 \
        --checkpoint.no-enable \
        --dump_folder "$out") 2>&1 | tail -50
    echo "[$(date)] $flavor seq=${SEQ_LEN} DONE"
}

# Order: paper dim=2304 first, reduce num_experts; then narrower dim variants.
for f in \
    kimi_linear_48b_block_attn_res_e32 \
    kimi_linear_48b_block_attn_res_e16 \
    kimi_linear_48b_block_attn_res_e8 \
    kimi_linear_48b_block_attn_res_d1280_e32 \
    kimi_linear_48b_block_attn_res_d1280_e16 \
    kimi_linear_48b_block_attn_res_d1024_e32 \
    kimi_linear_48b_block_attn_res_d1024_e16
do
    run_one "$f"
done

echo ""
echo "==============================================================="
echo "[$(date)] All sweep done."
echo "==============================================================="
