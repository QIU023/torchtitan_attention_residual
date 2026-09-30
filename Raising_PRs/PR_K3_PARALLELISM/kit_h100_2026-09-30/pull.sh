#!/bin/bash
# Pull the H100 results into this kit as they land (no traces, caches or checkpoints).
K=$(cd "$(dirname "$0")" && pwd); R="ssh -o ServerAliveInterval=15 -o ConnectTimeout=20 -p 18926"
for d in dep_h100_pp4vpp4 moonep_smoke moonep_rerun probe_step1; do
  rsync -a -e "$R" --exclude out --exclude "cache*" --exclude cc --exclude compile_trace --exclude "traces" --exclude "feats_*" root@115.124.123.240:mep/results/$d/ $K/results/$d/ 2>/dev/null
done
rsync -a -e "$R" root@115.124.123.240:mep/results/chain.txt $K/results/ 2>/dev/null
cat $K/results/dep_h100_pp4vpp4/progress.txt 2>/dev/null | tail -5
