#!/bin/bash
# Kimi K2.5 regression for this PR (it touches kimi_k2_7): dp2, tp=1, the debug config's own token
# budget (2048 per rank, 4096 per step; 256 or 512 per rank cannot hold one K2.5 multimodal row),
# 100 steps, this PR against main. K2.5 refuses tp > 1 on main (DistMuon, #3353) and cannot run on one GPU
# on main (DistMuon needs DTensor parameters; with AdamW, "Mesh 'loss' is not available"), so no seed
# checkpoint can be built: both cells start from --debug.seed 42. Waits for run_v2_dp2.sh.
set -u
until grep -q RUN-V2-DP2-DONE /workspace/tp_v2_dp2.log 2>/dev/null; do sleep 20; done
CFG=kimi_k2_5_debugmodel; MODULE=kimi_k2_7
. "$(dirname "$0")/common.sh"
cell k27_dp2_parent $TT_PARENT 0,1 2 k27 100 $D 2 &
cell k27_dp2        $TT        2,3 2 k27 100 $D 2 &
wait
echo
echo '# Kimi K2.5, dp2, tp=1, 4096 tokens per step (reference: main)'
table k27_dp2_parent k27_dp2
echo RUN-V2-K27-DONE
