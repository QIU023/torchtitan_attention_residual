#!/bin/bash
# Runs the groups added after the queue started, once the queue has finished.
until [ -f ~/k927/QUEUE_DONE ]; do sleep 30; done
bash ~/k927/kit/g6_dep_scaled.sh > ~/k927/g6.log 2>&1
echo done > ~/k927/AFTER_DONE
