#!/usr/bin/env python3
"""Create a seed checkpoint through the repo's own mechanism.

`create_seed_checkpoint` is a Trainer field marked tyro.conf.Suppress, so the
flag the older runners passed (`--checkpoint.create_seed_checkpoint`) no longer
exists on the CLI and main rejects it. scripts/_checkpoint_test_config.py is
what the repo's own comparison scripts use: it writes a temporary registry
module that loads the base config and sets the field. The temp module lives in
a TemporaryDirectory owned by this process, so the training run has to be a
child of it.

Usage: make_seed.py <module> <config> <dump_folder> [extra torchrun args...]
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.environ["TITAN"], "scripts"))
from _checkpoint_test_config import configure_checkpoint  # noqa: E402

module, config, dump = sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:]

env = os.environ.copy()
env.setdefault("PYTHONPATH", os.environ["TITAN"])
seed_module, seed_config = configure_checkpoint(env, module=module, config=config, mode="seed")

cmd = [
    "torchrun", "--nproc_per_node=1", "--master_port=30711",
    "-m", "torchtitan.train",
    "--module", seed_module, "--config", seed_config,
    "--dump-folder", dump,
] + extra
print("seed cmd:", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, env=env, cwd=os.environ["TITAN"]).returncode)
