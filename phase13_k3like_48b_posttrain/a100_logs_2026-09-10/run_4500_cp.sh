#!/bin/bash
# PR 4500 table on the A100: CP=1 (spmd_types) vs the two CP=2 recipes on the 4500 head, seed 42, 256 tokens, 100 steps, no seed checkpoint (her protocol).
source /workspace/venv/bin/activate; export PYTHONPATH=/workspace/attn_gym_up
OUT=/workspace/out_4500; mkdir -p $OUT; cd /workspace/wt_4500
COMMON="-m torchtitan.train --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 100 --training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
run() { local nm=$1 gpus=$2 np=$3; shift 3; CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm timeout 3000 torchrun --nproc_per_node=$np --master_port=$((31000+RANDOM%9000)) $COMMON "$@" --dump-folder $OUT/$nm > $OUT/$nm.log 2>&1; echo "$nm rc=$? steps=$(grep -a -c "step: " $OUT/$nm.log)"; }
run cp1 7 1 --module kimi_k3 --config kimi_k3_debugmodel --parallelism.data_parallel_shard_degree 1 --parallelism.spmd_backend spmd_types
run cp2ag 0,7 2 --module torchtitan_recipes.tests.b200 --config kimi_k3_debugmodel_mm_cp2 --parallelism.data_parallel_shard_degree 1
run cp2u 0,7 2 --module torchtitan_recipes.tests.b200 --config kimi_k3_debugmodel_mm_ulysses_cp2 --parallelism.data_parallel_shard_degree 1
python3 - <<"PY"
import re
def series(f):
    L,G={},{}
    for line in open(f, errors="ignore"):
        line=re.sub(r"\x1b\[[0-9;]*m","",line); m=re.search(r"step: *(\d+)\s.*loss: *([0-9.-]+).*grad_norm: *([0-9.]+)", line)
        if m: L[int(m.group(1))]=float(m.group(2)); G[int(m.group(1))]=float(m.group(3))
    return L,G
R=series("/workspace/out_4500/cp1.log")
print("| step | CP=1 loss | CP=2 all-gather (diff) | CP=2 Ulysses (diff) | CP=1 grad norm | all-gather gn (diff) | Ulysses gn (diff) |")
A=series("/workspace/out_4500/cp2ag.log"); U=series("/workspace/out_4500/cp2u.log")
for s in (1,2,5,10,20,50,100):
    if s not in R[0]: continue
    def d(X,i): return f"{X[i][s]:.6f} ({(X[i][s]-R[i][s])/R[i][s]*100:+.3f}%)" if s in X[i] else "-"
    print(f"| {s} | {R[0][s]:.6f} | {d(A,0)} | {d(U,0)} | {R[1][s]:.4f} | {d(A,1)} | {d(U,1)} |")
PY
echo CP4500-DONE
