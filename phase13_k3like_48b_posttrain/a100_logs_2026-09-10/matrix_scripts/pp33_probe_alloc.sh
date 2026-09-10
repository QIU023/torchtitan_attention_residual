#!/bin/bash
# The exact test: float32 model AND float32 experts (EXPERTS_FP32, a per-expert matmul loop in
# place of the bf16 grouped GEMM); dp1 against pp2 x vp4, step-1 gradients compared in float64.
# optimizer step. Runs on GPUs 6 and 7 beside the curve chain.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
T=/tmp/wt_ppprobe33x; SEED=$(ls -d /workspace/.mx3_seeds_main33/kimi_k3_debugmodel_*/seed_ckpt | head -1)
OUT=/workspace/ppprobe33y_$(date +%m%d_%H%M); mkdir -p $OUT; DUMP=$OUT/dumps; mkdir -p $DUMP
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
run(){ local nm=$1 np=$2 devs=$3; shift 3; local ALLOC=${ALLOC:-}
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  ( source /venv/main/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=$devs PYTHONPATH=$T TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
    PYTORCH_CUDA_ALLOC_CONF=${ALLOC:-} GRAD_TENSOR_DUMP=$DUMP/$nm GRAD_TENSOR_DUMP_EXIT=1 EXPERTS_FP32=1 timeout 2400 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps 1 $B --checkpoint.enable --checkpoint.interval 100000 $D "$@" \
    --dump-folder $d > $OUT/$nm.log 2>&1 ); rm -rf $d/checkpoint
  printf "%-10s rc=%s dumps=%s router=%s\n" $nm $? "$(ls $DUMP/$nm.rank*.pt 2>/dev/null | grep -vc router)" "$(ls $DUMP/$nm.rank*.router.pt 2>/dev/null | wc -l)"
  sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -m1 -iE "Traceback|Error|OutOfMemory" | cut -c1-160
}
run dp1 1 7
ALLOC=expandable_segments:True run dp1_alloc 1 7
run pp2_vp4 2 6,7 $P 2 $L 4 $IL
source /venv/main/bin/activate
python - "$DUMP" <<'PY'
import glob, sys, torch
D = sys.argv[1]
def load(prefix):
    d = {}
    for f in sorted(glob.glob(f"{D}/{prefix}.rank*.pt")):
        if f.endswith(".router.pt"): continue
        d.update(torch.load(f, map_location="cpu"))
    return d
def group(name):
    for key, g in (("tok_embeddings", "embedding"), ("lm_head", "head"), ("experts", "experts"), ("router", "router"),
                   ("shared", "shared experts"), ("delta_attention", "kda"), ("attention", "attention"), ("norm", "norms")):
        if key in name: return g
    return "other"
ga = load("dp1")
for other in ("dp1_alloc", "pp2_vp4"):
    gb = load(other); rels = []; groups = {}
    for k in ga:
        if k not in gb: continue
        x, y = ga[k].double(), gb[k].double(); r = ((x - y).norm() / (x.norm() + 1e-30)).item(); rels.append((r, k))
        g = groups.setdefault(group(k), [0.0, 0.0]); g[0] += float((x - y).pow(2).sum()); g[1] += float(x.pow(2).sum())
    rels.sort(); rs = [r for r, _ in rels]
    print(f"### float32 model + float32 experts, dp1 vs {other}: {len(rs)} parameters; rel L2 diff median {rs[len(rs)//2]:.2e} p90 {rs[int(0.9*len(rs))]:.2e} max {rs[-1]:.2e}; identical {sum(1 for r in rs if r == 0)}")
    print("  groups: " + ", ".join(f"{g} {((d2 / s2) ** 0.5 if s2 else 0):.1e}" for g, (d2, s2) in sorted(groups.items())))
PY
rm -rf $OUT/ind_* $OUT/tri_*
echo "PPPROBE33Y DONE $OUT"
