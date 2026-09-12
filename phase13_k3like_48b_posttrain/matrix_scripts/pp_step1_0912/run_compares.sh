#!/bin/bash
# Every step-1 comparison for the 24-layer debug model, 4 x 256 tokens.
source /workspace/venv_bfx9/bin/activate
C=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/cmp_pairs.py; D=/workspace/ppnum_0912
PP2=$(python3 -c "print(','.join(f'{l}:{0 if l<=11 else 1}' for l in range(24)))")
VP2=$(python3 -c "print(','.join(f'{l}:{0 if l<=5 else 1 if l<=12 else 2 if l<=18 else 3}' for l in range(24)))")
cmp(){ echo; echo "########## $1  vs  $2"; [ -n "$(ls $D/$1/g.ws* 2>/dev/null)" ] && [ -n "$(ls $D/$2/g.ws* 2>/dev/null)" ] || { echo "missing dump"; return; }; python $C $D/$1/g $D/$2/g $3; }
cmp dp1  dp1b
cmp dp1  pp2   $PP2
cmp dp1  vp2n  $VP2
cmp vp2n vp2c  $VP2
cmp dp1ns pp2  $PP2
cmp dp1ns vp2n $VP2
cmp dp1ns vp2c $VP2
cmp dp1  dp1ns
cmp vp2c vp2mut $VP2
