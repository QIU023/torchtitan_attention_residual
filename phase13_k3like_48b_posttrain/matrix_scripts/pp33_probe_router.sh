#!/bin/bash
# Do dp1 and pp2 x vp4 route the same tokens to the same experts at step 1? float32 model, the
# router's top-k ids recorded per call, gradients dumped in float32, the run left before the
# optimizer step. Runs on GPUs 6 and 7 beside the curve chain.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
T=/tmp/wt_ppprobe33r; SEED=$(ls -d /workspace/.mx3_seeds_main33/kimi_k3_debugmodel_*/seed_ckpt | head -1)
OUT=/workspace/ppprobe33r_$(date +%m%d_%H%M); mkdir -p $OUT; DUMP=$OUT/dumps; mkdir -p $DUMP
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
run(){ local nm=$1 np=$2 devs=$3; shift 3
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  ( source /venv/main/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=$devs PYTHONPATH=$T TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
    GRAD_TENSOR_DUMP=$DUMP/$nm GRAD_TENSOR_DUMP_EXIT=1 ROUTER_DUMP=1 timeout 2400 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps 1 $B --checkpoint.enable --checkpoint.interval 100000 $D "$@" \
    --dump-folder $d > $OUT/$nm.log 2>&1 ); rm -rf $d/checkpoint
  printf "%-10s rc=%s dumps=%s router=%s\n" $nm $? "$(ls $DUMP/$nm.rank*.pt 2>/dev/null | grep -vc router)" "$(ls $DUMP/$nm.rank*.router.pt 2>/dev/null | wc -l)"
  sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -m1 -iE "Traceback|Error|OutOfMemory" | cut -c1-160
}
run dp1 1 7
run pp2_vp4 2 6,7 $P 2 $L 4 $IL
source /venv/main/bin/activate
python - "$DUMP" <<'PY'
import glob, sys, torch, hashlib
D = sys.argv[1]
def load(prefix, suffix):
    d = {}
    for f in sorted(glob.glob(f"{D}/{prefix}.rank*{suffix}.pt")):
        if suffix == "" and f.endswith(".router.pt"): continue
        d.update(torch.load(f, map_location="cpu"))
    return d
ra, rb = load("dp1", ".router"), load("pp2_vp4", ".router")
print(f"routers: dp1 {len(ra)} pp2 {len(rb)}; calls per router dp1 {sorted({len(v) for v in ra.values()})} pp2 {sorted({len(v) for v in rb.values()})}")
# Calls are not aligned (selective AC recomputes the forward, the schedule reorders micro-batches,
# and the pipeline's shape inference adds calls), so routings are matched by content.
def sets(log):
    c = {}
    for t in log:
        c.setdefault(hashlib.sha1(torch.sort(t, dim=-1).values.numpy().tobytes()).hexdigest(), []).append(t)
    return c
tot_mb = 0; unmatched = 0; diff_tokens = 0; tot_tokens = 0; per = []
for k in sorted(ra):
    if k not in rb: continue
    A, B = sets(ra[k]), sets(rb[k])
    ua = [v[0] for h, v in A.items() if h not in B]; ub = [v[0] for h, v in B.items() if h not in A]
    tot_mb += len(A); unmatched += len(ua); dt = 0
    for a in ua:
        best = None
        for b in ub:
            if a.shape != b.shape: continue
            n = int((torch.sort(a, -1).values != torch.sort(b, -1).values).any(-1).sum())
            best = n if best is None or n < best else best
        dt += best if best is not None else a.shape[0]
    diff_tokens += dt; tot_tokens += sum(v[0].shape[0] for v in A.values())
    per.append((k, len(A), len(ua), dt, len(B)))
print(f"distinct micro-batch routings per router: dp1 {sorted({p[1] for p in per})}, pp2 {sorted({p[4] for p in per})}")
print(f"### routing dp1 vs pp2 x vp4 (matched by content): {unmatched} of {tot_mb} (router, micro-batch) routings have no identical twin; pairing the closest, {diff_tokens} of {tot_tokens} tokens differ ({100*diff_tokens/max(tot_tokens,1):.4f}%)")
for k, na, nu, dt, nb in per:
    if nu: print(f"  {k.split('._checkpoint')[0]}: {nu}/{na} routings differ, {dt} tokens")
ga, gb = load("dp1", ""), load("pp2_vp4", "")
rels = []
for k in ga:
    if k not in gb: continue
    x, y = ga[k].double(), gb[k].double(); rels.append(((x - y).norm() / (x.norm() + 1e-30)).item())
rels.sort(); print(f"### gradients (float32 model) dp1 vs pp2: rel L2 diff median {rels[len(rels)//2]:.2e} p90 {rels[int(0.9*len(rels))]:.2e} max {rels[-1]:.2e} over {len(rels)} parameters")
PY
for f in $DUMP/*.rank*.pt; do case $f in *.router.pt) ;; *) rm -f $f;; esac; done; rm -rf $OUT/ind_* $OUT/tri_*
echo "PPPROBE33R DONE $OUT"
