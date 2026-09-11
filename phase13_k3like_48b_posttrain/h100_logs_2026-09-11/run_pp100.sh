#!/bin/bash
export TITAN=/workspace/titan VENV=/venv/main PYPRE=/workspace/attn_gym_up
export SEED_ROOT=/workspace/.mx3_seeds_pp100
bash /workspace/pp100_controls.sh
touch /workspace/pp100.done
