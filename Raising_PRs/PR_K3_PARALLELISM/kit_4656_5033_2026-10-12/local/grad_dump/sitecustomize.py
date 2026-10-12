"""Logbook kit only. With GRAD_DUMP=<prefix>, every rank saves the local shard of each parameter's gradient at
optimizer step GRAD_DUMP_STEP (default 1; fp32, keyed by canonical name) to <prefix>.rank<N>.pt, to compare the
gradients of two cells at one step."""

import os

if os.environ.get("GRAD_DUMP") and "RANK" in os.environ:  # the torchrun parent has no RANK
    import torch
    from torch.distributed.tensor import DTensor
    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

    _target = int(os.environ.get("GRAD_DUMP_STEP", "1"))
    _steps: dict = {}
    _grads: dict = {}

    def _hook(optimizer, args, kwargs):
        groups = [g for g in optimizer.param_groups if g.get("param_names")]
        if not groups:
            return
        _steps[id(optimizer)] = _steps.get(id(optimizer), 0) + 1
        if _steps[id(optimizer)] != _target:
            return
        for group in groups:
            for name, param in zip(group["param_names"], group["params"]):
                if param.grad is None or name in _grads:
                    continue
                grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                _grads[name] = grad.detach().float().cpu().clone()

    def _post(optimizer, args, kwargs):
        if _steps.get(id(optimizer)) == _target and _grads:
            torch.save(_grads, f"{os.environ['GRAD_DUMP']}.rank{os.environ['RANK']}.pt")

    register_optimizer_step_pre_hook(_hook)
    register_optimizer_step_post_hook(_post)
