#!/bin/bash
# 1024 tokens, fp32 grad norm, the warmed shared cache (jitwarm_sh1024): the reference, pp2 and the two naive cells
# for 100 steps with the norm printed at full precision. Does the step-59 last-digit difference stay on one cache?
set -u; export CFG=kimi_k3_debugmodel_c4 GN_FP32=1 GN_REPR=1; . "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 $IL"
NAIVE="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"
J=$OUT/jitwarm_sh1024
cs() { local nm=$1 gpus=$2 np=$3 common=$4; shift 4; local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r $OUT/seed_c4_1024/checkpoint $d/
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$J/inductor TRITON_CACHE_DIR=$J/triton \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $common --training.steps 100 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  echo "$nm rc=$? gnrepr=$(grep -c '^GNREPR\|GNREPR ' $OUT/$nm.log)"; rm -rf $d/checkpoint; }
( export NOSYNC_GA=1; cs rp_dp1_ns 0 1 "$COMMON" $B4 $D 1 )
cs rp_pp2 0,1 2 "$COMMON" $B4 $D 1 $P2
cs rp_vp2n 0,1 2 "$NAIVE" $B4 $D 1 $P2 $IL
( export PP_STAGES_PER_RANK=4; cs rp_pp4vp4n 0,1,2,3 4 "$NAIVE" $B4 $D 1 $P4 )
python3 - $OUT <<'PY'
import re, sys
out = sys.argv[1]
def rd(nm):
    g = {}; L = {}
    for line in open(f"{out}/{nm}.log", errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"GNREPR (\d+) (\S+)", line)
        if m: g[int(m.group(1))] = float(m.group(2))
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.-]+)", line)
        if m and float(m.group(2)) > 0: L[int(m.group(1))] = float(m.group(2))
    return g, L
rg, rl = rd("rp_dp1_ns")
print("reference steps with a full-precision norm:", len(rg), "| step 59:", repr(rg.get(59)))
for nm in ("rp_pp2", "rp_vp2n", "rp_pp4vp4n"):
    g, L = rd(nm)
    dg = [(k, rg[k], g[k], (g[k]-rg[k])/rg[k]) for k in sorted(rg) if k in g and g[k] != rg[k]]
    dl = [k for k in sorted(rl) if L.get(k) != rl[k]]
    print(f"{nm}: norm differs at {len(dg)} steps, first {dg[:3]}; logged loss differs at {dl[:5]}")
PY
echo RUN-GNREPR-DONE
