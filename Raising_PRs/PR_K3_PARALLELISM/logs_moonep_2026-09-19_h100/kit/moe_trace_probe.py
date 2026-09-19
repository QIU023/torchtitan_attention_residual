"""Capture each MoE layer's router decisions and routed-expert input/output for one step.

Same body as grad_probe.py (a copy of Trainer.train_step truncated before the
optimizer step), plus forward hooks. Only the first call per module is kept, so
the activation-checkpoint recompute does not overwrite the forward.
"""
import os
import torch, torch.distributed as dist
from torch.distributed.tensor import DTensor
from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.distributed import utils as dist_utils

CAPTURE = {}


def _keep(t):
    if isinstance(t, DTensor):
        t = t.to_local()
    return t.detach().clone()


def _router_hook(name):
    def fn(mod, args, out):
        if name in CAPTURE:
            return
        scores_TK, ids_TK, routing_map_TE = out[0], out[1], out[2]
        CAPTURE[name] = {
            "x": _keep(args[0]),
            "scores": _keep(scores_TK),
            "ids": _keep(ids_TK),
            "counts": _keep(routing_map_TE).sum(dim=0),
        }
    return fn


def _experts_hook(name):
    def fn(mod, args, out):
        if name in CAPTURE:
            return
        CAPTURE[name] = {"x": _keep(args[0]), "out": _keep(out)}
    return fn


def _rows_hook(name):
    def fn(mod, args, out):
        if name in CAPTURE:
            return
        # args[1] is the per-row token count the dispatcher returned: a plain
        # tensor, unlike the plan, which is backed by symmetric memory.
        CAPTURE[name] = {"rows": _keep(args[1])}
    return fn


def main():
    init_logger()
    out_path = os.environ["MOE_TRACE_OUT"]
    config = ConfigManager().parse_args()
    sl.init_structured_logger(source="training", output_dir=config.dump_folder, enable=False)
    trainer = config.build()
    engine = trainer.engine
    engine.load_checkpoint()

    model = engine.model_parts[0]
    handles = []
    for name, mod in model.named_modules():
        short = name.replace("._checkpoint_wrapped_module", "")
        if short.endswith(".router"):
            handles.append(mod.register_forward_hook(_router_hook(short)))
        elif short.endswith(".routed_experts"):
            handles.append(mod.register_forward_hook(_experts_hook(short)))
        elif short.endswith(".inner_experts"):
            handles.append(mod.register_forward_hook(_rows_hook(short)))

    data_iterator = trainer.microbatch_generator(trainer.dataloader)
    microbatch_groups, local_valid_tokens = [], 0
    for _ in range(trainer.gradient_accumulation_steps):
        group = []
        for _ in range(trainer.num_pp_microbatches):
            microbatch = next(data_iterator)
            local_valid_tokens += microbatch.num_valid_tokens
            group.append(microbatch)
        microbatch_groups.append(group)

    pd = engine.parallel_dims
    local = torch.tensor(local_valid_tokens, dtype=torch.int64, device=engine.device)
    gvt = dist_utils.dist_sum_tensor(local, pd.get_mesh("dp")) if pd.dp_enabled else local
    gvt = engine.prepare_step(gvt, num_accumulation_steps=trainer.gradient_accumulation_steps)

    accumulated = None
    for index, group in enumerate(microbatch_groups):
        detached = engine.forward_backward_microbatch(
            microbatch_group=group, global_valid_tokens=gvt, accumulation_index=index
        )
        accumulated = detached.clone() if accumulated is None else accumulated.add_(detached)

    for h in handles:
        h.remove()

    if pd.dp_cp_enabled:
        lv = float(dist_utils.dist_sum(accumulated, pd.get_optional_mesh("loss")))
    else:
        lv = float(accumulated.item())

    if dist.get_rank() == 0:
        blob = {k: {kk: vv.float().cpu() for kk, vv in v.items()} for k, v in CAPTURE.items()}
        torch.save({"loss": lv, "capture": blob}, out_path)
        print(f"MOE_TRACE_OK {out_path} loss={lv:.6f} modules={len(blob)}", flush=True)
    trainer.close()
    dist.barrier(); dist.destroy_process_group()


main()
