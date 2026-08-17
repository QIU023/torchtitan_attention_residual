#!/usr/bin/env bash
# Overnight queue, 2026-08-17. One GPU job at a time -- the eight GPUs are one resource,
# and running two jobs concurrently has already produced a NaN that took six rounds to
# attribute to contention rather than to the code.
#
# Ordered by value, not by duration, because a queue that dies halfway should have done
# the important thing first. Each job is independent: a failure logs and the queue
# continues, since one unsupported configuration should not cost the rest of the night.
#
# Job 2 is the point of this run. "Does DEP's bubble scheduling hide anything" has never
# been answered -- every measurement so far is occupancy, and occupancy is not hiding.
# HANDOFF_2026-08-16 put the blocker at 60 GiB per GPU, from micro-batches >= stages with
# the micro-batch count equal to local_batch_size: pp8 x vp4 is 32 stages, so local batch
# 32, and seq 4096 x local 8 already OOMs in 15.5 GiB. That arithmetic is for 32 stages.
# At pp8 x vp2 it is 16 stages and local batch 16, halving the micro-batches in flight,
# and kimi_k3_debugmodel_bubble_ratio is already seq 4096 where the cost ratio is 0.493 --
# the point of maximum observable effect. So the queue walks down from the configuration
# theory says is best to ones that certainly fit, and takes the first that runs.
set -uo pipefail
: "${TITAN:?}"; : "${OUT:?}"
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate
HERE=${HERE:-/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts}

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/queue.log"; }
losses() { sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "step: +[0-9]+ +loss: +[0-9.]+"; }

# ----- 1: KCP parity at higher cp ------------------------------------------- #
# cp=2 splits the sequence in two, so the prefix scan composes exactly two fragments and
# every rank is either the first or the last. cp=4 and cp=8 are the first configurations
# with a MIDDLE rank, which is where a scan that mishandles composition shows up.
say "job 1: KCP forward+backward parity at cp=4 and cp=8"
for cp in 4 8; do
  gpus=$(seq -s, 0 $((cp - 1)))
  CUDA_VISIBLE_DEVICES=$gpus timeout 1800 torchrun --nproc_per_node="$cp" \
    --master_port=$((30100 + cp)) "$HERE/kcp_batch_parity.py" \
    > "$OUT/kcp_cp$cp.log" 2>&1
  say "  cp=$cp rc=$? -> $(grep -cE 'PARITY PASS' "$OUT/kcp_cp$cp.log") pass line(s); $(grep -oE 'worst [a-z_.0-9]+ [0-9.e-]+' "$OUT/kcp_cp$cp.log" | tail -1)"
done

# ----- 2: does the bubble hide anything -------------------------------------- #
# Step time with the mechanism off against on, at the honest cost ratio for the sequence
# each candidate runs. Occupancy counters are reported too, but they are not the judge.
say "job 2: DEP bubble step-time A/B, walking down from the theory-best shape"
# name:pp:layers_per_stage:local_batch:flavor:cost_ratio
CANDS=(
  "pp8vp2_seq4096:8:2:16:kimi_k3_debugmodel_bubble_ratio:0.493"
  "pp8vp2_seq2048:8:2:16:kimi_k3_debugmodel_report_arch_pp8vp4:2.0"
  "pp4vp2_seq4096:4:4:8:kimi_k3_debugmodel_bubble_ratio:0.493"
)
for cand in "${CANDS[@]}"; do
  IFS=: read -r name pp lps lb flavor ratio <<<"$cand"
  gpus=$(seq -s, 0 $((pp - 1)))
  ok=1
  for bub in 0 1; do
    tag="${name}_bub$bub"
    rm -rf "$OUT/$tag"
    KIMI_VIT_DEP=1 KIMI_VIT_DYNAMIC_CP=1 KIMI_VIT_BUBBLE=$bub \
      KIMI_VIT_BUBBLE_COST_RATIO=$ratio \
      CUDA_VISIBLE_DEVICES=$gpus timeout 5400 torchrun --nproc_per_node="$pp" \
      --master_port=30200 -m torchtitan.train \
      --module kimi_k3 --config "$flavor" --debug.seed 42 --debug.deterministic \
      --metrics.log_freq 1 --training.steps 20 \
      --training.global-batch-size "$lb" --training.local-batch-size "$lb" \
      --parallelism.data_parallel_shard_degree 1 \
      --parallelism.pipeline_parallel_degree "$pp" \
      --parallelism.pipeline_parallel_layers_per_stage "$lps" \
      --parallelism.pipeline_parallel_schedule Interleaved1F1B \
      --dump-folder "$OUT/$tag" > "$OUT/$tag.log" 2>&1
    [ "$(losses "$OUT/$tag.log" | wc -l)" -eq 0 ] && ok=0
    rm -rf "$OUT/$tag/checkpoint"
  done
  if [ "$ok" -eq 0 ]; then
    say "  $name: did not run -- $(grep -oiE '(OutOfMemoryError|CUDA out of memory|ValueError|must be)[^\"]{0,60}' "$OUT/${name}_bub0.log" "$OUT/${name}_bub1.log" 2>/dev/null | head -1)"
    continue
  fi
  # Median of the last 10 steps' tps: the first steps carry warmup and the cost ratio
  # question is about steady state.
  python3 - "$OUT/${name}_bub0.log" "$OUT/${name}_bub1.log" "$name" <<'PY' | tee -a "$OUT/queue.log"
import re, sys, statistics
def tps(p):
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(p).read())
    v = [float(x.replace(",", "")) for x in re.findall(r"tps: +([0-9,]+)", t)]
    return statistics.median(v[-10:]) if len(v) >= 10 else (statistics.median(v) if v else 0.0)
off, on = tps(sys.argv[1]), tps(sys.argv[2])
occ = re.findall(r"DEP bubble runtime: .{0,90}", re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[2]).read()))
gain = (on - off) / off * 100 if off else 0.0
print(f"  {sys.argv[3]}: tps off {off:.0f} -> on {on:.0f}  ({gain:+.2f}%)")
print(f"    {occ[-1] if occ else 'no bubble report'}")
print("    tps is the judge; the report line is only what the mechanism believes it did.")
PY
  break   # the first shape that runs is the one to report; the rest are fallbacks
done

# ----- 3: is the delta bias the same at a production shape? ------------------ #
# +0.19% was measured at 16 layers on two GPUs. The docstring says to re-check at a
# production shape before anyone weighs it against the memory saving.
say "job 3: delta-vs-naive bias at the 48B-layout carrier, pp8"
for mode in 0 1; do
  tag="bias48b_cache$mode"
  rm -rf "$OUT/$tag"
  TORCHTITAN_ATTNRES_CACHE=$mode CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 10800 \
    torchrun --nproc_per_node=8 --master_port=30300 -m torchtitan.train \
    --module kimi_k3 --config kimi_linear_48b_block_attn_res_d1280_e16_L32_N8 \
    --hf-assets-path ./tests/assets/tokenizer \
    --debug.seed 42 --debug.deterministic --metrics.log_freq 10 \
    --training.steps 400 --training.seq_len 512 \
    --training.global-batch-size 8 --training.local-batch-size 8 \
    --parallelism.data_parallel_shard_degree 1 \
    --parallelism.pipeline_parallel_degree 8 \
    --parallelism.pipeline_parallel_layers_per_stage 4 \
    --parallelism.pipeline_parallel_first_stage_less_layers 0 \
    --parallelism.pipeline_parallel_last_stage_less_layers 0 \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --dump-folder "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  rm -rf "$OUT/$tag/checkpoint"
done
w=$(grep -c "cache adapter wrapped" "$OUT/bias48b_cache1.log" || true)
say "  wrap lines on the delta arm: $w $([ "$w" -eq 0 ] && echo '(NOT delta mode -- comparison void)')"
python3 - "$OUT/bias48b_cache0.log" "$OUT/bias48b_cache1.log" <<'PY' | tee -a "$OUT/queue.log"
import re, sys, statistics
def curve(p):
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(p).read())
    seen = {}
    for s, l in re.findall(r"step: +(\d+) +loss: +([0-9.]+)", t):
        seen.setdefault(int(s), float(l))
    return [seen[k] for k in sorted(seen)]
off, on = curve(sys.argv[1]), curve(sys.argv[2])
n = min(len(off), len(on))
if n < 5:
    print(f"  too few points ({n})")
else:
    k = max(2, n // 10)
    a, b = statistics.fmean(off[-k:]), statistics.fmean(on[-k:])
    print(f"  48B-layout tail mean: naive {a:.5f} delta {b:.5f}  ({(b - a) / a * 100:+.3f}%)")
    print("  compare against +0.19% at 16 layers; a shape-dependent bias would show here.")
PY

# ----- 4: multi-commit at other geometries ---------------------------------- #
# pp2 x vp2 with two commits a stage is the only multi-commit shape verified. Four
# commits a stage, and a different pp, exercise the per-commit indexing harder.
say "job 4: multi-commit at pp2 x vp1-equivalent (4 commits/stage) and pp4"
for spec in "mc4:2:8" "mc_pp4:4:4"; do
  IFS=: read -r name pp lps <<<"$spec"
  gpus=$(seq -s, 0 $((pp - 1)))
  for mode in 0 1; do
    tag="${name}_cache$mode"
    rm -rf "$OUT/$tag"
    TORCHTITAN_ATTNRES_CACHE=$mode CUDA_VISIBLE_DEVICES=$gpus timeout 3600 torchrun \
      --nproc_per_node="$pp" --master_port=30400 -m torchtitan.train \
      --module kimi_k3 --config kimi_k3_mini_attnres_multicommit \
      --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
      --training.steps 6 --training.seq_len 256 \
      --training.global-batch-size 8 --training.local-batch-size 8 \
      --parallelism.data_parallel_shard_degree 1 \
      --parallelism.pipeline_parallel_degree "$pp" \
      --parallelism.pipeline_parallel_layers_per_stage "$lps" \
      --parallelism.pipeline_parallel_first_stage_less_layers 0 \
      --parallelism.pipeline_parallel_last_stage_less_layers 0 \
      --parallelism.pipeline_parallel_schedule Interleaved1F1B \
      --dump-folder "$OUT/$tag" > "$OUT/$tag.log" 2>&1
    rm -rf "$OUT/$tag/checkpoint"
  done
  w=$(grep -c "cache adapter wrapped" "$OUT/${name}_cache1.log" || true)
  same=$([ "$(losses "$OUT/${name}_cache0.log")" = "$(losses "$OUT/${name}_cache1.log")" ] && echo bitwise-equal || echo differs)
  bad=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/${name}_cache1.log" | grep -oE "capture-count mismatch|cleared [0-9]+ captured-grad slot" | sort -u | tr '\n' ' ')
  say "  $name: wrap=$w, $(losses "$OUT/${name}_cache1.log" | wc -l) loss lines, $same${bad:+, ANOMALY: $bad}"
done

# ----- 5: the gate --------------------------------------------------------- #
say "job 5: 58-cell gate"
OUT_GATE="$OUT/gate" bash -c "TITAN=$TITAN OUT=$OUT/gate bash $HERE/run_postmerge_gate.sh" \
  > "$OUT/gate_driver.log" 2>&1
say "  $(sed -n '/cell accounting/,$p' "$OUT/gate_driver.log" | grep -E 'logs found|passed' | head -1)"

say "OVERNIGHT QUEUE DONE"
