#!/bin/bash
# Dynamic CP (report sec 5.2.3) on the default configuration.
#
# One table, one run. There is no "off" flavor and none is needed: dynamic CP
# engages only at cp > 1, so this table's own dp1 row IS the disabled side --
# the same relationship every other matrix here uses.
#
# An earlier plan paired kimi_k3_debugmodel against a lowered-threshold flavor.
# Measured, both logged "Dynamic CP: 1 large image(s) of 1 over 1 sub-CP
# group(s) of 2 rank(s)": classify() is c >= min_patches and the debug image
# clears 256 as well as 64, so the pair would have compared a path against
# itself. The threshold flavor is gone and the default is what runs.
#
# seq 512: FlexAttention's BlockMask needs Q_LEN % (cp * 128) == 0, so cp4 needs
# at least 512, and one length keeps the cells on one curve.
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"
NB="--parallelism.context_parallel_load_balancer None"
# Cold KDA/tilelang compilation runs past the default 300s NCCL watchdog; that
# is a compile-cache artifact, not a hung collective.
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 512 --training.max-context-length 512 --comm.init-timeout-seconds 3600"
CELLS="dp1|1|$D 1
cp2|2|$D 1 $C 2 $NB
cp4|4|$D 1 $C 4 $NB"

TITAN=/workspace/tt_rebase CFG=kimi_k3_debugmodel BATCH="$B" CELLS="$CELLS" $MX mm_dyncp
echo "DYNAMIC CP MATRIX DONE"
