"""tp_trainer_grad_probe with TF32 disabled.

The dp1 runs store parameters in fp32, but torch.backends.cuda.matmul
fp32_precision defaults to tf32 on this GPU -- 10 mantissa bits, comparable to
bf16 -- so "fp32" runs are not high precision in the matmuls at all. Forcing
true fp32 separates a placement defect (survives) from arithmetic noise
(collapses).
"""

from __future__ import annotations

import runpy
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
except Exception:
    pass

import os
import runpy

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp_trainer_grad_probe.py"),
    run_name="__main__",
)
