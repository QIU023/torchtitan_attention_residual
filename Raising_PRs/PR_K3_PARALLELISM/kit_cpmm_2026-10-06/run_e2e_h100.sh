#!/bin/bash
# #4380 end-to-end with large images on the H100, after the tower benchmark (kit cpmm_probe.py cells via run_matrix2.sh).
# A: numerics. Debug model, cc12m-test resized to 1008 px (72 x 72 patches, so every micro-batch splits its image),
#    CP2 all-gather, typechecking on and AdamW as in the body table; main twice, the PR, and the PR with main's vision
#    bank swapped in; 20 steps (the 32-sample set holds one image per micro-batch, no sample repeats). GPUs 0,1.
# B: training-level memory. The released K3 tower (27 layers, dim 1024) on the debug text model, 2016 px
#    (144 x 144 patches, seq 8192), recipe path (typechecking off: compile + SAC); main and the PR at CP2 on GPUs 2,3,
#    then at CP4 on all four; 10 steps.
K=/workspace/kit_cpmm; M=/workspace/tt_main; C=/workspace/tt_cpmm; O=/workspace/h100_e2e
export KIT_ENV=/workspace/kit_setup/env_cu126.sh PP_PRE=/workspace/kit_setup/shims WARM_STEPS=5
mkdir -p $O
A="CP_MODE=allgather,CP_IMG_PX=1008"
STEPS=20 BANK_SWAP_LOG=$O/a/swaplog $K/run_matrix2.sh $O/a \
  main:$M:0,1:$A,WARM=a main_b:$M:0,1:$A,WARM=a pr:$C:0,1:$A,WARM=a swap:$C:0,1:$A,BANK_SWAP=1,WARM=a &
B="CP_MODE=allgather,CP_IMG_PX=2016,CP_SEQ=8192,CP_TOWER=k3,CP_TYPECHECK=0"
STEPS=10 $K/run_matrix2.sh $O/b2 main:$M:2,3:$B,WARM=b2 pr:$C:2,3:$B,WARM=b2 &
wait
STEPS=10 $K/run_matrix2.sh $O/b4 main:$M:0,1,2,3:$B,CP_DEGREE=4,WARM=b4 pr:$C:0,1,2,3:$B,CP_DEGREE=4,WARM=b4
echo E2E_DONE > $O/done.txt
