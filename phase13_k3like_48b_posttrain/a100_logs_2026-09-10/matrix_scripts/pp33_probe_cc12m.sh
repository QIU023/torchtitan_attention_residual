#!/bin/bash
# The exact test on streamed cc12m (variable padding per micro-batch), where the step-1 loss of pp2 read
# 12.35294 against dp1's 12.35295: float32 model and experts, routing recorded, gradients dumped;
# is the head's gradient bitwise, is the routing, and what is the layer profile.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
T=/tmp/wt_ppprobe33x; SEED=$(ls -d /workspace/.mx3_seeds_main33/kimi_k3_debugmodel_*/seed_ckpt | head -1)
OUT=/workspace/ppprobe33c_$(date +%m%d_%H%M); mkdir -p $OUT; DUMP=$OUT/dumps; mkdir -p $DUMP
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
run(){ local nm=$1 np=$2 devs=$3; shift 3
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  ( source /venv/main/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=$devs PYTHONPATH=$T TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
    GRAD_TENSOR_DUMP=$DUMP/$nm GRAD_TENSOR_DUMP_EXIT=1 ROUTER_DUMP=1 EXPERTS_FP32=1 timeout 2400 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_cc12m --debug.seed 42 --debug.deterministic \
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
rels.sort(); print(f"### gradients (float32 model + experts) dp1 vs pp2 on cc12m: rel L2 diff median {rels[len(rels)//2]:.2e} p90 {rels[int(0.9*len(rels))]:.2e} max {rels[-1]:.2e} over {len(rels)} parameters")
by_layer = {}; groups = {}
for k in ga:
    if k not in gb: continue
    x, y = ga[k].double(), gb[k].double(); d2 = float((x - y).pow(2).sum()); s2 = float(x.pow(2).sum())
    g = "head" if "lm_head" in k else ("output_res" if "output_res" in k else ("embedding" if "tok_embeddings" in k else ("vision" if "vision" in k else "layers")))
    gg = groups.setdefault(g, [0.0, 0.0]); gg[0] += d2; gg[1] += s2
    if k.startswith("layers."):
        li = int(k.split(".")[1]); b = by_layer.setdefault(li, [0.0, 0.0]); b[0] += d2; b[1] += s2
print("  groups: " + ", ".join(f"{g} {((d2 / s2) ** 0.5 if s2 else 0):.1e}" for g, (d2, s2) in sorted(groups.items())))
print("  by layer: " + " ".join(f"{li}:{((d2 / s2) ** 0.5 if s2 else 0):.0e}" for li, (d2, s2) in sorted(by_layer.items())))
PY
for f in $DUMP/*.rank*.pt; do case $f in *.router.pt) ;; *) rm -f $f;; esac; done; rm -rf $OUT/ind_* $OUT/tri_*
echo "PPPROBE33C DONE $OUT"
