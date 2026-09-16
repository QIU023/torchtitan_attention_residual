"""Tower cut vs replicated on the same weights and image, on the tree under test.

torchrun --nproc_per_node=2 tower_parity_probe.py  (PYTHONPATH: scratchpad, attn gym, the tree)

Builds the allgather-KV cp2 recipe through the Trainer (so the model is parallelized with the cp mesh and the
sub-groups), takes the first micro-batch's image, and encodes it twice: partitioned across the pair (threshold 96)
and replicated (threshold 1e9). Prints the max abs difference over the scale of the merged tokens, forward only,
in the model's dtype and in fp32 (the tower cast to fp32 for the second pair).
"""
import os

import torch
import torch.distributed as dist

from cpmm_probe import kimi_k3_mm_allgather_kv_cp2_min96
from torchtitan.distributed.spmd_types import spmd_local_context
from torchtitan.trainer import Trainer


def main() -> None:
    cfg = kimi_k3_mm_allgather_kv_cp2_min96()
    cfg.training.steps = 1
    cfg.dump_folder = os.environ["DUMP"]
    cfg.debug.seed = 42
    cfg.debug.deterministic = True
    if os.environ.get("PROBE_FP32") == "1":
        # fp32 compute end to end: FSDP's mixed precision casts params to bf16 in forward otherwise.
        cfg.training.dtype = "float32"
        cfg.training.mixed_precision_param = "float32"
        cfg.training.mixed_precision_reduce = "float32"
    tr = Trainer(cfg)
    model = tr.model_parts[0]
    batch = next(iter(tr.dataloader))
    inputs = batch[0] if isinstance(batch, tuple) else batch
    pv, grid = inputs["pixel_values"].to("cuda"), inputs["grid_thw"].to("cuda")
    rank = dist.get_rank()
    if rank == 0:
        print("PROBE grids", grid.tolist(), flush=True)
    out = {}
    with torch.no_grad(), spmd_local_context("dp"):
        for name, thr in (("cut", 96), ("rep", 10**9)):
            model.dynamic_cp_min_patches = thr
            model._dyncp_logged = True
            out[name] = model.encode_images(pv, grid).float()
    d = (out["cut"] - out["rep"]).abs().max().item()
    scale = out["rep"].abs().max().item()
    if rank == 0:
        print(f"PROBE tower ({'fp32' if os.environ.get('PROBE_FP32') == '1' else 'bf16'} compute, weight dtype {next(model.vision_encoder.parameters()).dtype}): max|cut-rep|={d:.3e} scale={scale:.3e} ratio={d / scale:.3e} shape={tuple(out['rep'].shape)}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
