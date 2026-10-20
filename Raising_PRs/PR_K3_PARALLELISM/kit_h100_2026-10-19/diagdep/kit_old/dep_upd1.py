"""DEP diagnosis (logbook kit only): dep_new_local's cells; at the first optimizer step every rank saves its own
parameters' gradients before it and the parameters after it (DUMP_DIR/{grad1,param1}_rank<r>.pt) and prints the
clipping norm in full precision."""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import dep_new_local


def _save(engine, tag: str) -> None:
    out = {}
    for part in engine.model_parts:
        for name, param in part.named_parameters():
            t = param.grad if tag == "grad1" else param
            if t is None:
                continue
            t = t.full_tensor() if isinstance(t, DTensor) else t
            out[name.replace("._checkpoint_wrapped_module", "")] = t.detach().cpu()
    os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
    torch.save(out, f"{os.environ['DUMP_DIR']}/{tag}_rank{dist.get_rank()}.pt")
    print(f"DEP_UPD1_DUMP {tag} rank {dist.get_rank()} {len(out)} tensors", flush=True)


def _patch() -> None:
    from torchtitan.training_engine import TrainingEngine

    original = TrainingEngine.optim_step

    def optim_step(self):
        first = self.num_completed_steps == 0
        if first:
            _save(self, "grad1")
        grad_norm = original(self)
        if first:
            print(f"DEP_UPD1_GRAD_NORM rank {dist.get_rank()} {float(grad_norm)!r}", flush=True)
            _save(self, "param1")
        return grad_norm

    TrainingEngine.optim_step = optim_step
    print(f"[dep_upd1] rank {os.environ.get('RANK')} optim_step patched", flush=True)


def w_dep_off():
    _patch()
    return dep_new_local.w_dep_off()


def w_dep_k25():
    _patch()
    return dep_new_local.w_dep_k25()


def w_dep_bubble():
    _patch()
    return dep_new_local.w_dep_bubble()
