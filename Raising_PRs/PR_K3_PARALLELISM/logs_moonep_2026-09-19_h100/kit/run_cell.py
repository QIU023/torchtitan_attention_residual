#!/usr/bin/env python3
"""Run one matrix cell with checkpointing enabled, through the repo's mechanism.

The checkpoint.* CLI group is gone on current main; scripts/_checkpoint_test_config.py
is what the repo's own comparison scripts use to enable the optional checkpointer
for an arbitrary registry config. mode="load" turns it on and marks it load-only,
which is what a cell wants: read the staged seed, write nothing back.

Usage: NPROC=<n> run_cell.py <module> <config> <dump_folder> [extra flags...]
"""
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.environ["TITAN"], "scripts"))
from _checkpoint_test_config import configure_checkpoint  # noqa: E402

module, config, dump = sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:]

env = os.environ.copy()
env.setdefault("PYTHONPATH", os.environ["TITAN"])
cell_module, cell_config = configure_checkpoint(env, module=module, config=config, mode="load")

cmd = [
    "torchrun",
    f"--nproc_per_node={os.environ.get('NPROC', '1')}",
    f"--master_port={30000 + random.randrange(20000)}",
    "-m", "torchtitan.train",
    "--module", cell_module, "--config", cell_config,
    "--dump-folder", dump,
] + extra
print("cell cmd:", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, env=env, cwd=os.environ["TITAN"]).returncode)
