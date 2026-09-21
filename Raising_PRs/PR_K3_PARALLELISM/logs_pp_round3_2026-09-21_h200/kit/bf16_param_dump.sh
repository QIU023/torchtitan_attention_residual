#!/bin/bash
# bf16, matched reference vs pp2 x vp2 cache off, 20 steps on the bf16 lineage cache, fp32 master parameters dumped at
# step 20 (PARAM_DUMP_STEP) and the total norm bits every step: do the masters differ by a unit while the printed loss agrees?
set -eu
TT=/workspace/tt_pp; OUT=/workspace/results/pp_r3_bf16dump; SRC=/workspace/results/pp_r3; LOG=/workspace/results/pp_r3_run.log; K=/workspace/kit; mkdir -p $OUT
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT NCCL_NVLS_ENABLE=0 GN_FP32=1 GN_REPR=1 PARAM_DUMP_STEP=20
cd $TT && python3 - <<'PY'
import pathlib
p = pathlib.Path("torchtitan/training_engine.py"); s = p.read_text()
old = 'if __import__("os").environ.get("PARAM_DUMP") and current_step == 1:'
new = 'if __import__("os").environ.get("PARAM_DUMP") and current_step == int(__import__("os").environ.get("PARAM_DUMP_STEP", "1")):'
if old in s: s = s.replace(old, new); p.write_text(s); print("PARAM_DUMP_STEP added")
else: assert new in s, "PARAM_DUMP hack not found"; print("PARAM_DUMP_STEP present")
PY
python3 -m py_compile torchtitan/training_engine.py
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 20"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"; P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
cell() {  # cell <name> <gpus> <nproc> <port> <config> <flags...>
  local nm=$1 gpus=$2 np=$3 port=$4 cfg=$5; shift 5
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $SRC/seed_c4_1024/checkpoint $d/
  cp -r $SRC/ind_dp1_ns_seqsum $OUT/ind_$nm; cp -r $SRC/tri_dp1 $OUT/tri_$nm
  set +e
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm PARAM_DUMP=$OUT/$nm \
      torchrun --nproc_per_node=$np --master_port=$port $COMMON --config $cfg "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; set -e; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) params=$(ls $OUT/params_$nm.rank*.pt 2>/dev/null | wc -l) $(date -u +%T)" >> $LOG
}
echo "== BD: bf16 master parameters at step 20, reference vs pp2 x vp2 cache off $(date -u +%T)" >> $LOG
( export NOSYNC_GA=1; cell ref 0 1 47501 kimi_k3_debugmodel_c4 $B4 ) &
cell vp2n 2,3 2 47502 kimi_k3_debugmodel_c4_pp_naive $B4 $P2 $IL &
wait
cd $OUT && python3 - >> $OUT/compare.txt 2>&1 <<'PY'
import torch, glob, re
def load(nm):
    d = {}
    for f in sorted(glob.glob(f"params_{nm}.rank*.pt")): d.update(torch.load(f))
    return d
a, b = load("ref"), load("vp2n")
common = sorted(set(a) & set(b)); diff_t = 0; diff_el = 0; total = 0; max_ulp = 0; bf_flips = 0
for n in common:
    x, y = a[n].float(), b[n].float(); total += x.numel()
    ne = (x != y)
    if ne.any():
        diff_t += 1; diff_el += int(ne.sum())
        ulp = ((x.view(torch.int32) - y.view(torch.int32)).abs()[ne]).max().item(); max_ulp = max(max_ulp, ulp)
    bf_flips += int((x.to(torch.bfloat16) != y.to(torch.bfloat16)).sum())
print(f"fp32 master parameters at step 20: {diff_t}/{len(common)} tensors differ, {diff_el}/{total} elements, max distance {max_ulp} fp32 units; bf16 copies differing: {bf_flips} elements")
def norms(nm):
    out = {}
    for f in glob.glob(f"{nm}.log"):
        for l in open(f, errors="ignore"):
            m = re.search(r"GNREPR r0 (\d+) (\S+) (0x\S+)", l)
            if m: out[int(m.group(1))] = (m.group(2), m.group(3))
    return out
na, nb = norms("ref"), norms("vp2n")
d = [s for s in sorted(na) if s in nb and na[s][1] != nb[s][1]]
print(f"total norm bits differ on steps {d} of {len(na)}; e.g. step {d[0] if d else '-'}: {na[d[0]] if d else ''} vs {nb[d[0]] if d else ''}")
def losses(nm):
    return {int(m.group(1)): m.group(2) for l in open(f"{nm}.log", errors="ignore") for m in [re.search(r"step: *(\d+) .*loss: *([0-9.]+)", re.sub(r"\x1b\[[0-9;]*m", "", l))] if m}
la, lb = losses("ref"), losses("vp2n")
print("printed loss identical on", sum(1 for s in la if lb.get(s) == la[s]), "of", len(la), "steps")
PY
cat $OUT/compare.txt >> $LOG; echo "BD-DONE $(date -u +%T)" >> $LOG
