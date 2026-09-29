#!/bin/bash
# Copy the box's results into the logbook kit (logs, rc, tables, memory records, trace analyses;
# no compile caches, dumps, traces or tensors).
K=$(cd "$(dirname "$0")" && pwd); mkdir -p $K/results
rsync -a -e "ssh -p 28444 -o ConnectTimeout=15" --exclude "cache*" --exclude "out/" --exclude "dump/" \
  --exclude "*.pt" --exclude "*.pkl" --exclude "traces/" --exclude "*trace.json*" \
  root@45.135.56.11:mep/results/ $K/results/ 2>&1 | grep -v "Welcome\|Have fun\|AI agents"
du -sh $K/results
