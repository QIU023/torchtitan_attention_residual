# Shared by the run scripts. Set TT (branch tree), TT_PARENT (main worktree), OUT.
# Two GPUs: every cell runs to completion before the next one starts.
TT=${TT:-$PWD}; TT_PARENT=${TT_PARENT:-$PWD/../tt_parent}; OUT=${OUT:-$PWD/tp_h100_out}; mkdir -p $OUT
CFG=${CFG:-kimi_k3_debugmodel}
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"; E="--parallelism.expert_parallel_degree"
NOSP="--parallelism.no-enable-sequence-parallel"; TC="--debug.spmd_typechecking"  # spmd_backend is gone on main after #4419; type checking needs TAIL=activation_checkpoint:none (SAC + flex is refused)
COMMON="-m torchtitan.train --module ${MODULE:-kimi_k3} --config $CFG --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --checkpoint.enable --checkpoint.interval 100000"

seed() {  # seed <tag> <batch flags...>: one seed checkpoint per batch shape, built on the branch tree with 1 GPU
  local tag=$1; shift
  if [ ! -d $OUT/seed_$tag/checkpoint ]; then
    ( cd $TT && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
        $COMMON --training.steps 1 --checkpoint.create_seed_checkpoint "$@" --dump-folder $OUT/seed_$tag > $OUT/seed_$tag.log 2>&1 )
    echo "seed_$tag rc=$?"
  fi
}

cell() {  # cell <name> <tree> <gpus> <nproc> <seed tag> <steps> <flags...>
  local nm=$1 tree=$2 gpus=$3 np=$4 stag=$5 steps=$6; shift 6
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; [ -d $OUT/seed_$stag/checkpoint ] && cp -r $OUT/seed_$stag/checkpoint $d/  # no seed dir: fresh init from --debug.seed
  ( cd $tree && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm TRITON_CACHE_BASE=$OUT/tri_$nm \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
      $COMMON --training.steps $steps "$@" --dump-folder $d ${TAIL:-} > $OUT/$nm.log 2>&1 )  # TAIL: a positional subcommand goes last
  local rc=$?; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log)"
}

table() {  # table <reference name> <names...>: loss and grad norm at TABLE_STEPS (default "1 10 20")
  python3 - "$OUT" "$@" <<'PY'
import os, re, sys
out, ref, names = sys.argv[1], sys.argv[2], sys.argv[3:]
STEPS = tuple(int(x) for x in os.environ.get("TABLE_STEPS", "1 10 20").split())
def series(f):
    L, G = {}, {}
    for line in open(f, errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.-]+).*grad_norm: *([0-9.]+)", line)
        if m and float(m.group(2)) > 0:
            L[int(m.group(1))] = float(m.group(2)); G[int(m.group(1))] = float(m.group(3))
    return L, G
R = series(f"{out}/{ref}.log")
hdr = " | ".join(f"step {s}" for s in STEPS)
print(f"| cell | {hdr} | {hdr} (grad norm) |")
print("|---" * (1 + 2 * len(STEPS)) + "|")
for nm in [ref] + names:
    try:
        L, G = series(f"{out}/{nm}.log")
    except FileNotFoundError:
        print(f"| {nm} | missing |"); continue
    def fmt(X, Y, s):
        if s not in X: return "-"
        if nm == ref or s not in Y: return f"`{X[s]:.6f}`"
        if X[s] == Y[s]: return f"`{X[s]:.6f}` (bitwise)"
        return f"`{X[s]:.6f}` ({(X[s]-Y[s])/Y[s]*100:+.3g}%)"
    print(f"| {nm} | " + " | ".join(fmt(L, R[0], s) for s in STEPS)
          + " | " + " | ".join(fmt(G, R[1], s) for s in STEPS) + " |")
PY
}
