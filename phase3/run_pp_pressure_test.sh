#!/usr/bin/env bash
# PP × VP pressure test — multi-config sweep, naive vs adapter.
#
# Trains from scratch on C4 for STEPS=$STEPS (default 1000) at each
# (config, PP, VP) combo, with the cross-stage caching adapter ON
# vs OFF. Outputs are kept side-by-side so loss curves can be
# diffed and step times averaged.
#
# Designed to be carrier-realistic but training cost is minimal
# (random init, 1k steps, 175M with extra depth).
#
# Grid (default — override with SWEEP="config:PP:VP[:µbs:gbs] ..."):
#
#   175M_L32_n8 PP=8 VP=4   (32 chunks, 1 layer/chunk — aggressive)
#   175M_L32_n8 PP=4 VP=8   (32 chunks, same total but higher VP)
#   175M_L48_n8 PP=8 VP=6   (48 chunks, 1 layer/chunk — prod-depth)
#   175M_L16_n8 PP=8 VP=2   (16 chunks, sanity vs Phase-3 history)
#
# Each combo runs naive then adapter; results go to
# phase3/runs/pressure_test_${TIMESTAMP}/{config}_pp{P}_vp{V}_{naive,adapter}/
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$SCRIPT_DIR/.." && pwd)"
TORCHTITAN_DIR="$WS/torchtitan"

STEPS="${STEPS:-1000}"
NGPU="${NGPU:-8}"
TIMESTAMP="$(date +%Y%m%d-%H%M)"
SWEEP_OUT_ROOT="${SWEEP_OUT_ROOT:-$SCRIPT_DIR/runs/pressure_test_${TIMESTAMP}}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

# Sweep: space-separated "config:pp:vp:lbs:gbs" tuples.
# lbs/gbs control microbatch sizing; defaults below satisfy
# num_microbatches = gbs / (DP * lbs) >= PP * VP.
# Config function names are snake_case lowercase. Tuple = config:PP:VP:LBS:GBS.
#
# LBS must satisfy n_microbatches = LBS / pipeline_parallel_microbatch_size
# >= num_total_stages = PP * VP. Default pipeline_parallel_microbatch_size = 1,
# so LBS >= PP * VP. Otherwise the schedule deadlocks in batch_isend_irecv
# at _step_microbatches:1730 (rather than erroring up front).
# Historical phase3 used LBS=4 which masked this — that was less than
# PP*VP=16 for the pp8_adapter run, so it likely had a real bubble or
# incorrect schedule shape too.
#
# Sweep at PP=8 VP=2 across three depths. PP=8 VP=4 needs LBS=32 which
# is large but valid; add as a stretch row.
# Tuple = config:PP:VP:LBS:GBS.
# Constraints:
#   1. n_microbatches = LBS / pipeline_parallel_microbatch_size >= PP*VP
#      (microbatch_size default 1, so LBS >= PP*VP).
#   2. GBS must be divisible by LBS * DP where DP = NGPU/PP.
#      e.g. PP=4 on NGPU=8 -> DP=2, so GBS = LBS * 2 * k.
#      PP=8 on NGPU=8 -> DP=1, so GBS = LBS * k.
SWEEP="${SWEEP:-\
175m_attn_res_L16_n8:8:2:16:16 \
175m_attn_res_L16_n8:4:2:8:16 \
175m_attn_res_L16_n8:4:4:16:32 \
}"

mkdir -p "$SWEEP_OUT_ROOT"
SUMMARY="$SWEEP_OUT_ROOT/SUMMARY.md"
cat > "$SUMMARY" <<EOF
# PP Pressure Test — ${TIMESTAMP}

steps=${STEPS} ngpu=${NGPU}

| config | PP | VP | LBS | GBS | mode | avg step time (s) | final loss | out dir |
|---|---|---|---|---|---|---|---|---|
EOF

RUN_NAIVE="${RUN_NAIVE:-1}"
RUN_ADAPTER="${RUN_ADAPTER:-1}"

run_one() {
    local cfg="$1" pp="$2" vp="$3" lbs="$4" gbs="$5" mode="$6"
    # Derive layers_per_stage from carrier depth.
    # L<n>_n<b> → n = num_layers. virtual_stages = num_layers/layers_per_stage
    # VP = virtual_stages / PP, so layers_per_stage = num_layers / (PP * VP).
    local n_layers
    n_layers=$(echo "$cfg" | grep -oE "_L[0-9]+_" | grep -oE "[0-9]+")
    if [[ -z "$n_layers" ]]; then n_layers=16; fi  # legacy 175m_attn_res has 16
    local layers_per_stage=$(( n_layers / (pp * vp) ))
    if [[ $layers_per_stage -lt 1 ]]; then layers_per_stage=1; fi
    local run_name="${cfg}_pp${pp}_vp${vp}_${mode}"
    local out_dir="$SWEEP_OUT_ROOT/$run_name"
    mkdir -p "$out_dir"
    echo "$(cd "$TORCHTITAN_DIR" && git rev-parse --short HEAD)" > "$out_dir/GIT_SHA"

    local cache_arg=""
    if [[ "$mode" == "adapter" ]]; then
        cache_arg="TORCHTITAN_ATTNRES_CACHE=1"
    fi

    echo ""
    echo "==============================================================="
    echo "[$(date)] $run_name STEPS=$STEPS LBS=$lbs GBS=$gbs"
    echo "==============================================================="

    local lr_arg=""
    if [[ -n "${LR:-}" ]]; then
        lr_arg="--optimizer.lr $LR"
    fi
    local warmup_arg=""
    if [[ -n "${WARMUP:-}" ]]; then
        warmup_arg="--lr_scheduler.warmup_steps $WARMUP --lr_scheduler.total_steps $STEPS"
    fi

    (cd "$TORCHTITAN_DIR" && \
     env $cache_arg ATTNRES_DBG=0 \
         PYTORCH_ALLOC_CONF="expandable_segments:True" \
         torchrun \
             --nproc_per_node="$NGPU" \
             --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
             --local-ranks-filter 7 --role rank --tee 3 \
             -m torchtitan.train \
             --module attn_res --config "llama3_${cfg}" \
             --training.steps "$STEPS" \
             --training.local_batch_size "$lbs" \
             --training.global_batch_size "$gbs" \
             $lr_arg $warmup_arg \
             --parallelism.pipeline_parallel_degree "$pp" \
             --parallelism.pipeline_parallel_schedule "Interleaved1F1B" \
             --parallelism.pipeline_parallel_layers_per_stage "$layers_per_stage" \
             --parallelism.pipeline_parallel_first_stage_less_layers 0 \
             --parallelism.pipeline_parallel_last_stage_less_layers 0 \
             --checkpoint.no-enable \
             --dump_folder "$out_dir" \
             --metrics.save_tb_folder tb \
             > "$out_dir/train.log" 2>&1)
    local rc=$?

    # Extract avg step time from steady-state tps (skip first 50 warmup)
    local avg_step
    avg_step=$(awk '/step:/{n+=1; if(n>50){c+=1; for(i=1;i<=NF;i++)if($i~/^step:/){idx=i+1; t=$(idx); split(t,parts,":");}}} END{print c}' "$out_dir/train.log" 2>/dev/null || echo "?")
    local final_loss
    final_loss=$(grep -aoE "loss:\s+[0-9.]+" "$out_dir/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$" || echo "?")

    # Compute step time from log timestamps over the last 100 steps
    local step_time
    step_time=$(python3 - <<PYEOF
import re, sys
ts = []
try:
    for line in open("$out_dir/train.log"):
        m = re.search(r"(\d{2}:\d{2}:\d{2},\d{3}).*step:\s*(\d+)", line)
        if m:
            ts.append(m.group(1))
    if len(ts) < 50:
        print("?"); sys.exit()
    # Use last 50 timestamps
    last = ts[-50:]
    from datetime import datetime
    parsed = [datetime.strptime(t.replace(",", "."), "%H:%M:%S.%f") for t in last]
    deltas = [(parsed[i+1]-parsed[i]).total_seconds() for i in range(len(parsed)-1)]
    if deltas:
        print(f"{sum(deltas)/len(deltas):.2f}")
    else:
        print("?")
except Exception:
    print("?")
PYEOF
    )

    echo "| $cfg | $pp | $vp | $lbs | $gbs | $mode | $step_time | $final_loss | $run_name |" >> "$SUMMARY"
    echo "[$(date)] $run_name done rc=$rc step=$step_time loss=$final_loss"
}

# Loop the sweep
for tuple in $SWEEP; do
    IFS=":" read -r cfg pp vp lbs gbs <<< "$tuple"
    if [[ -z "${gbs:-}" ]]; then
        echo "skip malformed: $tuple"; continue
    fi
    if [[ "$RUN_NAIVE" == "1" ]]; then
        run_one "$cfg" "$pp" "$vp" "$lbs" "$gbs" "naive"
    fi
    if [[ "$RUN_ADAPTER" == "1" ]]; then
        run_one "$cfg" "$pp" "$vp" "$lbs" "$gbs" "adapter"
    fi
done

echo ""
echo "PRESSURE TEST DONE. Summary at $SUMMARY"
cat "$SUMMARY"
