"""Step-1 gradients for the #4380 H100 table (logbook kit only): cpmm_probe's cell for one step; every parameter's full
gradient before the first optimizer step (so before clipping) in DUMP_DIR/grad1_rank<r>.pt."""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import cpmm_probe


def _dump(engine) -> None:
    out = {}
    for part in engine.model_parts:
        for name, param in part.named_parameters():
            g = param.grad
            if g is None:
                continue
            g = g.full_tensor() if isinstance(g, DTensor) else g
            out[name.replace("._checkpoint_wrapped_module", "")] = g.detach().cpu()
    os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
    torch.save(out, f"{os.environ['DUMP_DIR']}/grad1_rank{dist.get_rank()}.pt")
    print(f"GRAD1_DUMP rank {dist.get_rank()} {len(out)} tensors", flush=True)


def cell():
    from torchtitan.training_engine import TrainingEngine

    original = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0:
            _dump(self)
        return original(self)

    TrainingEngine.optim_step = optim_step
    print(f"[grad1_probe] rank {os.environ.get('RANK')} optim_step patched", flush=True)
    return cpmm_probe.cell()
