#!/bin/bash
# The c4 1024-token stream and the 256-token kit again with the total grad norm in float32 (GN_FP32=1,
# gn_fp32_hack.py): clipping fires every step (max_norm 1.0, norms 1.5-23), and with bf16 gradients the bf16 norm
# depends on the PP partition. Same seeds; OUT for 1024 = run_pp_c4.sh's, OUT_R64 = the 256 kit's.
set -u; export CFG=kimi_k3_debugmodel_c4 GN_FP32=1; . "$(dirname "$0")/common.sh"
OUT_R64=${OUT_R64:-/workspace/out_c4r64}
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 $IL"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
( export NOSYNC_GA=1; cell gn32_c4_dp1_ns $TT 0 1 c4_1024 100 $B4 $D 1 ) &
( export NOSYNC_GA=1 MB_REVERSE=1; cell gn32_c4_dp1_ns_rev $TT 1 1 c4_1024 100 $B4 $D 1 ) &
cell gn32_c4_pp2 $TT 2,3 2 c4_1024 100 $B4 $D 1 $P &
wait
cell gn32_c4_vp2c $TT 0,1 2 c4_1024 100 $B4 $D 1 $P $IL &
( naive_common; cell gn32_c4_vp2n $TT 2,3 2 c4_1024 100 $B4 $D 1 $P $IL ) &
wait
( export PP_STAGES_PER_RANK=4; cell gn32_c4_pp4vp4c $TT 0,1,2,3 4 c4_1024 100 $B4 $D 1 $P4 )
( export PP_STAGES_PER_RANK=4; naive_common; cell gn32_c4_pp4vp4n $TT 0,1,2,3 4 c4_1024 100 $B4 $D 1 $P4 )
export TABLE_STEPS="1 2 3 5 10 20 100"
echo; echo '# c4 1024, grad norm in fp32, reference dp1 no-sync (fp32 norm)'
table gn32_c4_dp1_ns gn32_c4_dp1_ns_rev gn32_c4_pp2 gn32_c4_vp2c gn32_c4_vp2n gn32_c4_pp4vp4c gn32_c4_pp4vp4n
OUT=$OUT_R64; export C4_ROW_TOKENS=64
B4="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 64"
( export NOSYNC_GA=1; cell gn32_r64_dp1_ns $TT 0 1 c4_256 100 $B4 $D 1 ) &
cell gn32_r64_pp2 $TT 2,3 2 c4_256 100 $B4 $D 1 $P &
wait
( naive_common; cell gn32_r64_vp2n $TT 0,1 2 c4_256 100 $B4 $D 1 $P $IL ) &
cell gn32_r64_vp2c $TT 2,3 2 c4_256 100 $B4 $D 1 $P $IL &
wait
echo; echo '# 256 tokens, grad norm in fp32, reference dp1 no-sync (fp32 norm)'
table gn32_r64_dp1_ns gn32_r64_pp2 gn32_r64_vp2n gn32_r64_vp2c
echo RUN-PP-C4-GN32-DONE
