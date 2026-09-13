#!/bin/bash
# The 256-token kit's interleaved cells read 1 bf16 ulp above dp1 in the step-1 grad norm (23.125 vs 23.000) with or
# without the cache, while pp2 is bitwise. On one shared cache (5060) the naive interleaved cell is 680/680 bitwise at
# step 1, so rerun them here on a copy of the reference's own inductor / triton cache. OUT = the 256 kit's OUT.
set -u; export CFG=kimi_k3_debugmodel_c4 C4_ROW_TOKENS=64; . "$(dirname "$0")/common.sh"
B4="--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 64"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
naive_common() { COMMON="${COMMON/--config kimi_k3_debugmodel_c4 /--config kimi_k3_debugmodel_c4_pp_naive }"; }
warm() { rm -rf $OUT/ind_$1 $OUT/tri_$1; cp -r $OUT/ind_c4r64_dp1_ns $OUT/ind_$1; cp -r $OUT/tri_c4r64_dp1_ns $OUT/tri_$1; }
warm c4r64_vp2n_warm; ( naive_common; cell c4r64_vp2n_warm $TT 0,1 2 c4_256 100 $B4 $D 1 $P $IL ) &
warm c4r64_vp2c_warm; cell c4r64_vp2c_warm $TT 2,3 2 c4_256 100 $B4 $D 1 $P $IL &
wait
export TABLE_STEPS="1 2 3 10 20 100"
echo; echo '# 256 tokens, reference dp1 no-sync; _warm = on a copy of the reference cache'
table c4r64_dp1_ns c4r64_pp2 c4r64_vp2n c4r64_vp2n_warm c4r64_vp2c c4r64_vp2c_warm c4r64_pp4vp4n c4r64_pp4vp4c
echo RUN-PP-C4-256-WARM-DONE
