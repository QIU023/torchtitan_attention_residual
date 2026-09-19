#!/usr/bin/env python3
"""Run a probe script (grad_probe.py, row_occupancy_probe.py) with checkpointing on.

Same reason as run_cell.py: the checkpoint.* CLI group is gone, so the probes
that used to pass --checkpoint.enable have to go through the repo's helper.

Usage: NPROC=<n> PORT=<p> run_probe.py <probe.py> <module> <config> <dump> [flags...]
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.environ["TITAN"], "scripts"))
from _checkpoint_test_config import configure_checkpoint  # noqa: E402

probe, module, config, dump = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
extra = sys.argv[5:]

env = os.environ.copy()
env.setdefault("PYTHONPATH", os.environ["TITAN"])
pm, pc = configure_checkpoint(env, module=module, config=config, mode="load")

cmd = [
    "torchrun", f"--nproc_per_node={os.environ.get('NPROC', '1')}",
    f"--master_port={os.environ.get('PORT', '41599')}",
    probe, "--module", pm, "--config", pc, "--dump-folder", dump,
] + extra
print("probe cmd:", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, env=env, cwd=os.environ["TITAN"]).returncode)
