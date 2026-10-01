#!/bin/bash
# 10-01 night: phase 2b after phase 2a: DEP off / K2.5 / bubble with traces at q19 (seq 8192, 448 px, cap 1, cost
# ratio 0.19 from phase 1), M16, through phase2_dep.sh.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
until [ -f $S/dep_1001/phase2a/DONE ]; do sleep 20; done
LEVELS="q19:8192:1:0.19:16:448" OUT=$S/dep_1001/phase2b bash /workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_overnight_2026-10-01/phase2_dep.sh
