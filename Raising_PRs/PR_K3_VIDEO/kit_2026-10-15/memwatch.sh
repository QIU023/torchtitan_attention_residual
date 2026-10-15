#!/bin/bash
# Kill the big-input training processes if the container's memory passes the limit (GiB). Logs every 30 s.
LIMIT=${1:-200}; LOG=$2
while true; do
  cur=$(awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current)
  echo "$(date +%H:%M:%S) ${cur}GiB" >> $LOG
  if [ "$cur" -ge "$LIMIT" ]; then
    for p in $(pgrep -f "torchtitan.train --module big"); do kill $p; done
    echo "$(date +%H:%M:%S) KILLED at ${cur}GiB" >> $LOG
  fi
  sleep 30
done
