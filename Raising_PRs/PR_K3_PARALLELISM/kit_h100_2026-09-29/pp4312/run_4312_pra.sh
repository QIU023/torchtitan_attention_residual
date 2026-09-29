#!/bin/bash
# PR 4312's H100 matrix for PR A on 4 x H100: main 5dc97a3e7 against PR A 85eefa54b with its receive-buffer
# overrides removed (the send waits kept; PR A as committed fails on this torch), both with the matrix's probe
# hacks, on the source-built 09-28 torch. The 4312 protocol: c4 text rows, seed 42, deterministic, one seed
# checkpoint per batch shape, total grad norm in fp32, 100 steps; the dp1 reference accumulates like the
# pipeline (NOSYNC_GA), runs cold to fill the caches and again on a copy, and every cell copies that cache.
# Table A: 1024 tokens per step (4 x 256), dp1. Table C: 2048 tokens per step, dp2.
M=~/mep; K=~/kit/pp4312; OUT=$M/results/pra4312; mkdir -p $OUT; P=$OUT/progress.txt
. $M/venv_src/bin/activate
A=$M/w/main_5dc9; B=$M/w/pra_send
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
for t in $A $B; do python $K/probe_apply_5dc9.py $t >> $P 2>&1 || { note "probe failed on $t"; exit 1; }; done
note "main $(git -C $A rev-parse --short HEAD) pra_send $(git -C $B rev-parse --short HEAD)+probe torch $(python -c 'import torch; print(torch.__version__)')"
export GN_FP32=1 NCCL_NVLS_ENABLE=0 PYTHONPATH=$K
COMMON="-m torchtitan.train --module pp4312_local --debug.seed 42 --debug.deterministic --metrics.log_freq 1"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
D="--parallelism.data_parallel_shard_degree"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4"
C4=kimi_k3_debugmodel_c4; C4N=kimi_k3_debugmodel_c4_pp_naive
C8=kimi_k3_debugmodel_c4_8stages; C8N=kimi_k3_debugmodel_c4_8stages_naive
C16=kimi_k3_debugmodel_c4_16stages; C16N=kimi_k3_debugmodel_c4_16stages_naive
seed() {  # seed <tag> <batch flags...>
  local tag=$1; shift
  ( cd $A && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$OUT/ind_seed_$tag TRITON_CACHE_DIR=$OUT/tri_seed_$tag \
      torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
      $COMMON --config kimi_k3_debugmodel_c4_seed --training.steps 1 "$@" --dump-folder $OUT/seed_$tag > $OUT/seed_$tag.log 2>&1 )
  note "seed_$tag rc=$? checkpoint=$(ls -d $OUT/seed_$tag/checkpoint 2>/dev/null | wc -l)"
}
cell() {  # cell <name> <tree> <gpus> <seed tag> <cache source or -> <config> <flags...>
  local nm=$1 tree=$2 gpus=$3 stag=$4 csrc=$5 cfg=$6; shift 6
  local d=$OUT/$nm np=$(echo $gpus | tr ',' '\n' | wc -l); rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d
  cp -r $OUT/seed_$stag/checkpoint $d/
  if [ "$csrc" != "-" ]; then cp -r $OUT/ind_$csrc $OUT/ind_$nm; cp -r $OUT/tri_$csrc $OUT/tri_$nm 2>/dev/null; fi
  ( cd $tree && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
      timeout 2400 torchrun --nproc_per_node=$np --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $COMMON --config $cfg --training.steps 100 "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; rm -rf $d
  note "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $OUT/$nm.log)"
}
pair() {  # pair <name> <seed tag> <ref> <config> <flags...>: main on GPUs 0-1 and PR A on 2-3 at once
  local nm=$1 stag=$2 ref=$3 cfg=$4; shift 4
  cell main_$nm $A 0,1 $stag $ref $cfg "$@" & cell pra_$nm $B 2,3 $stag $ref $cfg "$@" & wait
  rm -rf $OUT/ind_main_$nm $OUT/tri_main_$nm $OUT/ind_pra_$nm $OUT/tri_pra_$nm
}
quad() {  # quad <name> <seed tag> <ref> <config> <flags...>: main, then PR A, on all four GPUs
  local nm=$1 stag=$2 ref=$3 cfg=$4; shift 4
  cell main_$nm $A 0,1,2,3 $stag $ref $cfg "$@"; cell pra_$nm $B 0,1,2,3 $stag $ref $cfg "$@"
  rm -rf $OUT/ind_main_$nm $OUT/tri_main_$nm $OUT/ind_pra_$nm $OUT/tri_pra_$nm
}
seed c4_1024 $B4
seed c4_2048 $B8
# Table A: the reference (cold fill, then the warm rerun that is the reference), dp1 rows, then every PP cell
NOSYNC_GA=1 cell ref_cold $A 0 c4_1024 - $C4 $B4 $D 1
NOSYNC_GA=1 cell ref $A 0 c4_1024 ref_cold $C4 $B4 $D 1
cell main_dp1 $A 1 c4_1024 ref $C4 $B4 $D 1 & MB_REVERSE=1 cell main_dp1_rev $A 2 c4_1024 ref $C4 $B4 $D 1 &
NOSYNC_GA=1 cell main_cold2 $A 3 c4_1024 - $C4 $B4 $D 1 & wait
pair pp2 c4_1024 ref $C4 $B4 $D 1 $P2
pair vp2c c4_1024 ref $C4 $B4 $D 1 $P2 $IL
pair vp2n c4_1024 ref $C4N $B4 $D 1 $P2 $IL
pair pp2vp4c c4_1024 ref $C8 $B4 $D 1 $P2 $IL
pair pp2vp4n c4_1024 ref $C8N $B4 $D 1 $P2 $IL
quad pp4 c4_1024 ref $C4 $B4 $D 1 $P4
quad pp4vp2c c4_1024 ref $C4 $B4 $D 1 $P4 $IL
quad pp4vp2n c4_1024 ref $C4N $B4 $D 1 $P4 $IL
quad pp4vp4c c4_1024 ref $C16 $B4 $D 1 $P4 $IL
quad pp4vp4n c4_1024 ref $C16N $B4 $D 1 $P4 $IL
note "table A done"
# Table C: dp2, 2048 tokens per step, its own seed, reference and cache
NOSYNC_GA=1 cell d2_ref_cold $A 0,1 c4_2048 - $C4 $B8 $D 2
NOSYNC_GA=1 cell d2_ref $A 0,1 c4_2048 d2_ref_cold $C4 $B8 $D 2
quad d2_pp2 c4_2048 d2_ref $C4 $B8 $D 2 $P2
quad d2_vp2c c4_2048 d2_ref $C4 $B8 $D 2 $P2 $IL
quad d2_vp2n c4_2048 d2_ref $C4N $B8 $D 2 $P2 $IL
note "table C done"
python $K/tables_pra4312.py $OUT ref cold2 dp1 dp1_rev pp2 pp4 vp2n vp2c pp2vp4n pp2vp4c pp4vp2n pp4vp2c pp4vp4n pp4vp4c > $OUT/tables_A.md 2>&1
python $K/tables_pra4312.py $OUT d2_ref d2_pp2 d2_vp2n d2_vp2c > $OUT/tables_C.md 2>&1
rm -rf $OUT/ind_* $OUT/tri_* $OUT/seed_*/checkpoint
for t in $A $B; do git -C $t checkout -- torchtitan/models/kimi_k3/model.py torchtitan/trainer.py torchtitan/training_engine.py torchtitan/distributed/utils.py; done
for t in $A $B; do echo "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | tr '\n' ' ')"; done > $OUT/trees_after.txt
note "pra4312 done"; echo done > $OUT/DONE
