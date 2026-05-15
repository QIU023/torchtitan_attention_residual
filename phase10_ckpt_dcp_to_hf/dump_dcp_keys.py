"""Dump DCP checkpoint keys + shapes for mapping design.

Single-rank DCP load to inspect what's in the phase4 ckpt. Run via:
    torchrun --nproc_per_node=1 phase10_ckpt_dcp_to_hf/dump_dcp_keys.py <ckpt-dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS / "torchtitan"))


def init_dist():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir", type=Path)
    p.add_argument("--config", default="kimi_linear_436m_block_attn_res_n4")
    args = p.parse_args()

    init_dist()
    torch.cuda.set_device(0)

    from torchtitan.experiments.kimi_linear.config_registry import (
        kimi_linear_436m_block_attn_res_n4,
    )
    from torchtitan.experiments.kimi_linear.attn_res_model import (
        KimiLinearAttnResModel,
    )

    cfg = kimi_linear_436m_block_attn_res_n4()
    spec = cfg.model_spec.model
    print(f"# spec.kimi_config:")
    for k in dir(spec.kimi_config):
        if not k.startswith("_"):
            v = getattr(spec.kimi_config, k)
            if isinstance(v, (int, float, str, bool, list, tuple)):
                print(f"#   {k}={v!r}")

    model = KimiLinearAttnResModel(spec.kimi_config, num_blocks=spec.num_blocks)
    model.init_weights()
    sd = model.state_dict()

    print(f"\n# total params (from skeleton): {len(sd)}")
    print(f"# loading DCP ckpt from {args.ckpt_dir}")
    dcp.load(sd, checkpoint_id=str(args.ckpt_dir))

    # Group by prefix
    print("\n# === all params (name | shape | dtype) ===")
    for k in sorted(sd.keys()):
        v = sd[k]
        print(f"{k}\t{tuple(v.shape)}\t{v.dtype}")

    print(f"\n# total: {sum(v.numel() for v in sd.values()):,} params")


if __name__ == "__main__":
    main()
