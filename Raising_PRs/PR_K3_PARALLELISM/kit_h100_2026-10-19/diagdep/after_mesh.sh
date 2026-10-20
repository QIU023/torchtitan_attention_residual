#!/bin/bash
D=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/diag4380
until grep -q "MESH_DONE" $D/progress_mesh.txt 2>/dev/null; do sleep 10; done
exec /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/diagdep/run_diagdep.sh
