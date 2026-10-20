"""DEP diagnosis (logbook kit only): dep_new_local's DEP-off cell; on rank 0, every post-accumulate call of the stage-0
tower's unsharded FSDP parameters records the gradient it leaves (with grad_dtype set this is the running sum), saved at
the first optimizer step to DUMP_DIR/ownergrad_rank0.pt as {fqn: [tensor per call]}."""
import os

import torch
import torch.distributed as dist

import dep_new_local


def _patch() -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    from torchtitan.training_engine import TrainingEngine

    calls: dict = {}
    original_init = FSDPParam.init_unsharded_param

    def init_unsharded_param(self, *args, **kwargs):
        had = getattr(self, "_unsharded_param", None) is not None
        result = original_init(self, *args, **kwargs)
        param = getattr(self, "_unsharded_param", None)
        fqn = getattr(self, "_param_fqn", "")
        if not had and param is not None and "vision_encoder" in fqn and dist.get_rank() == 0:
            param.register_post_accumulate_grad_hook(
                lambda p, fqn=fqn: calls.setdefault(fqn, []).append(p.grad.detach().clone().cpu())
            )
        return result

    FSDPParam.init_unsharded_param = init_unsharded_param
    original_step = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0 and dist.get_rank() == 0:
            os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
            torch.save(calls, f"{os.environ['DUMP_DIR']}/ownergrad_rank0.pt")
            lens = sorted({len(v) for v in calls.values()})
            dtypes = sorted({str(t.dtype) for v in calls.values() for t in v})
            print(f"DEP_OWNERGRAD params {len(calls)} calls per param {lens} dtypes {dtypes}", flush=True)
        return original_step(self)

    TrainingEngine.optim_step = optim_step
    print(f"[dep_ownergrad] rank {os.environ.get('RANK')} patched", flush=True)


def w_dep_off():
    _patch()
    return dep_new_local.w_dep_off()
