#!/bin/bash
# After chain3 (DEP and the tower PR column): the #4380 cells' step-1 gradients.
C=/workspace/chain.txt
until grep -q "CHAIN3_DONE" $C; do sleep 30; done
echo "$(date +%H:%M:%S) grad1 starts" >> $C
/workspace/kit_cpmm/run_grad1.sh
echo "$(date +%H:%M:%S) CHAIN4_DONE $(tail -2 /workspace/h100_grad1/progress.txt | head -1)" >> $C
