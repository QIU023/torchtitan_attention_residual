#!/bin/bash
# Groups C and D of the 5060 smokes: every rank's peak at step 5, identity against the #4656 cell, balance spread.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; R=$S/s5060
KL=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_pp_lowerbound_2026-09-26
/workspace/venv_bfx9/bin/python $KL/tab_lb.py $R 5 c_b4656 c_pra c_o4765all c_b4764plan
echo
/workspace/venv_bfx9/bin/python $KL/balance_table.py $R 5 d_off d_bal d_planbal
