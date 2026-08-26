#!/bin/bash
# Dynamic CP taken to the last degree this box can run.
#
# A separate table from the seq-512 one, not an extra row on it: cp8 needs
# Q_LEN % (cp * 128) == 0, so 1024, and one table is one run at one length.
# Every degree is re-measured here against this table's own dp1.
#
# The question it settles: the seq-512 table has cp2 at 2.6e-4 and cp4 at
# 3.5e-3, a 13x rise over one doubling, while text-side CP saturates from cp4 to
# cp8. Two degrees cannot tell a trend from a single point.
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"
NB="--parallelism.context_parallel_load_balancer None"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 1024 --training.max-context-length 1024 --comm.init-timeout-seconds 3600"
CELLS="dp1|1|$D 1
cp2|2|$D 1 $C 2 $NB
cp4|4|$D 1 $C 4 $NB
cp8|8|$D 1 $C 8 $NB"

TITAN=/workspace/tt_depfix CFG=kimi_k3_debugmodel BATCH="$B" CELLS="$CELLS" $MX mm_dyncp8
echo "DYNAMIC CP seq1024 MATRIX DONE"
