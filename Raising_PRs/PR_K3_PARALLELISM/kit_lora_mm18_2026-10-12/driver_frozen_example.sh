#!/bin/bash
# Frozen-stage optimizer cells on three trees: without the optimizer commit, the PR head, and the empty-rank port.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; A=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mm18_1012; F=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mm18_1012/frozen; K=$A/kit3
export ENTRYPOINT=mm18_train_export
bash $K/run_frozen.sh $S/wt_lora $F/pr $F/seed "pp4_vpp2|pp=4,vpp=2|0,1,2,3" "pp8|pp=8|0,1,2,3,4,5,6,7"
bash $K/run_frozen.sh $S/wt_lora_noopt $F/noopt $F/seed "pp4_vpp2|pp=4,vpp=2|0,1,2,3"
bash $K/run_frozen.sh $S/wt_lora_opt $F/opt $F/seed "pp8|pp=8|0,1,2,3,4,5,6,7" "pp4_vpp2|pp=4,vpp=2|0,1,2,3"
echo done > $F/DONE
