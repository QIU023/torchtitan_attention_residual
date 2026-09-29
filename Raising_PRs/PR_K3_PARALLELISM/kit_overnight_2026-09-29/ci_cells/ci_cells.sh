#!/bin/bash
# The Kimi K3 cells of the B200 suite on 8 x 5060 (torch 2.15.0.dev20260928): PR A ec8bb420a against the rebased
# #4656 4ae9422db, recipe as committed (unseeded), each on its own cache: the PP cell (8 GPUs), mm (4), mm_muon (2).
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
V=/workspace/venv_0928; R=$S/ci_cells; P=$R/progress.txt; mkdir -p $R
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
cell() {  # <name> <tree> <config> <ngpu>
  local name=$1 tree=$2 cfg=$3 n=$4; local D=$R/$name; mkdir -p $D/cache
  ( cd $tree && CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) TORCHINDUCTOR_CACHE_DIR=$D/cache/ic TRITON_CACHE_DIR=$D/cache/tc \
    timeout 2400 $V/bin/torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module torchtitan_recipes.tests.b200 --config $cfg --metrics.log-freq 1 --dump-folder $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out $D/cache
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -o 'step: *[0-9]* *loss: *[0-9.]*' | tail -1) $(grep -a -m1 -o 'Error: .*' $D/run.log | cut -c1-160)"
}
note "pra $(git -C $S/wt_pra_rb rev-parse --short HEAD) 4656rb $(git -C $S/wt_4656_rb rev-parse --short HEAD)"
for t in pra:$S/wt_pra_rb 4656rb:$S/wt_4656_rb; do
  cell ${t%%:*}_pp ${t#*:} kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4 8
  cell ${t%%:*}_mm ${t#*:} kimi_k3_debugmodel_mm 4
  cell ${t%%:*}_mm_muon ${t#*:} kimi_k3_debugmodel_mm_muon 2
done
note "ci cells done"
