"""DEP diagnosis (logbook kit only): dep_new_local's cells; every rank records its replica's tower gradient for each
micro-batch it backpropagates (post-accumulate hooks, before DEP adds it to its fp32 buffer) and, at the first
optimizer step, saves those (DUMP_DIR/mbgrad_rank<r>.pt) with the tower's final gradients (DUMP_DIR/grad1_rank<r>.pt)."""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import dep_new_local


def _patch() -> None:
    from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.runtime import VisionDep
    from torchtitan.training_engine import TrainingEngine

    captured: dict = {}
    current = [None]
    hooked = [False]
    original_backward = VisionDep._backward

    def backward(self, mb):
        if not hooked[0]:
            for name, param in zip(self._names, self._params):
                param.register_post_accumulate_grad_hook(
                    lambda p, name=name: captured.__setitem__((current[0], name), p.grad.detach().clone().cpu())
                )
            hooked[0] = True
        current[0] = mb
        return original_backward(self, mb)

    VisionDep._backward = backward
    original_step = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0:
            out = {}
            for part in self.model_parts:
                for name, param in part.named_parameters():
                    if "vision_encoder" in name and param.grad is not None:
                        g = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
                        out[name.replace("._checkpoint_wrapped_module", "")] = g.detach().cpu()
            os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
            r = dist.get_rank()
            torch.save(out, f"{os.environ['DUMP_DIR']}/grad1_rank{r}.pt")
            torch.save(dict(captured), f"{os.environ['DUMP_DIR']}/mbgrad_rank{r}.pt")
            print(f"DEP_MBGRAD rank {r} micro-batches {sorted({mb for mb, _ in captured})} final tower grads {len(out)}", flush=True)
        return original_step(self)

    TrainingEngine.optim_step = optim_step
    print(f"[dep_mbgrad] rank {os.environ.get('RANK')} patched", flush=True)


def w_dep_k25():
    _patch()
    return dep_new_local.w_dep_k25()
