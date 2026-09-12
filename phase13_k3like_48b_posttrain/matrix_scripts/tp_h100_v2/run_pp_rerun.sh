#!/bin/bash
# Reruns after the KDA probe switch was first put on bound_gate (every cell started after that edit
# died at step 0): the dp2 stream, the KDA-autotune-off pass, and the bf16-reduce reversed floor.
set -u
cd "$(dirname "$0")"
bash run_pp_dp2.sh > /workspace/pp_dp2_run.log 2>&1
bash run_pp_kdanoat.sh > /workspace/pp_kdanoat_run.log 2>&1
( . ./common.sh
  R="--training.mixed_precision_reduce bfloat16"
  B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
  ( export MB_REVERSE=1; cell br_dp1_rev $TT 0 1 bf16_1024 100 $B4 $R $D 1 )
  echo; echo '# bf16 reduce, reversed accumulation against dp1'; table br_dp1 br_dp1_rev ) > /workspace/pp_bf16r_rev.log 2>&1
echo RUN-PP-RERUN-DONE
