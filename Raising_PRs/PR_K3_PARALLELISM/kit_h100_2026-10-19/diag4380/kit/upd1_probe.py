"""#4380 diagnosis (logbook kit only): cpmm_probe's cell; at the first optimizer step, every parameter's full gradient
before it and every parameter after it (rank 0 saves DUMP_DIR/{grad1,param1}.pt), and the clipping norm in full
precision. CLIP_OFF=1 raises max_norm so clipping never scales."""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import cpmm_probe


def _full(engine, tag: str) -> None:
    out = {}
    for part in engine.model_parts:
        for name, param in part.named_parameters():
            t = param.grad if tag == "grad1" else param
            if t is None:
                continue
            t = t.full_tensor() if isinstance(t, DTensor) else t
            out[name.replace("._checkpoint_wrapped_module", "")] = t.detach().cpu()
    if dist.get_rank() == 0:
        os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
        torch.save(out, f"{os.environ['DUMP_DIR']}/{tag}.pt")
        print(f"UPD1_DUMP {tag} {len(out)} tensors", flush=True)


def cell():
    from torchtitan.training_engine import TrainingEngine

    original = TrainingEngine.optim_step

    def optim_step(self):
        first = self.num_completed_steps == 0
        if first:
            _full(self, "grad1")
        grad_norm = original(self)
        if first:
            print(f"UPD1_GRAD_NORM {float(grad_norm)!r}", flush=True)
            _full(self, "param1")
        return grad_norm

    TrainingEngine.optim_step = optim_step
    config = cpmm_probe.cell()
    if os.environ.get("CLIP_OFF") == "1":
        config.optim.max_norm = 1.0e9
    print(f"[upd1_probe] rank {os.environ.get('RANK')} patched, max_norm {config.optim.max_norm}", flush=True)
    return config


def seed():
    return cpmm_probe.seed()
