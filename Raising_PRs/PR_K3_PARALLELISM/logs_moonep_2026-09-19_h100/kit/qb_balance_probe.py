"""Routing imbalance and per-rank load after N real training steps.

Quantile balancing moves the expert bias in the optimizer pre-hook, so it does
nothing at step 1; this runs full steps (optimizer included) and captures the
last one. Reports maxvio, the report's axis, and the per-rank computed rows.
"""
import json
import os

import torch, torch.distributed as dist
from torch.distributed.tensor import DTensor
from torchtitan.config import ConfigManager
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger

CAPTURE = {}
ARMED = {"on": False}


def _keep(t):
    if isinstance(t, DTensor):
        t = t.to_local()
    return t.detach().clone()


def _router_hook(name):
    def fn(mod, args, out):
        if not ARMED["on"] or (name, "counts") in CAPTURE:
            return
        CAPTURE[(name, "counts")] = _keep(out[2]).sum(dim=0)
    return fn


def _rows_hook(name):
    def fn(mod, args, out):
        if not ARMED["on"] or (name, "rows") in CAPTURE:
            return
        CAPTURE[(name, "rows")] = _keep(args[1])
        CAPTURE[(name, "E")] = int(getattr(mod, "num_experts", 0))
    return fn


def main():
    init_logger()
    steps = int(os.environ.get("PROBE_STEPS", 20))
    out_path = os.environ["QB_PROBE_OUT"]
    config = ConfigManager().parse_args()
    sl.init_structured_logger(source="training", output_dir=config.dump_folder, enable=False)
    trainer = config.build()
    engine = trainer.engine
    engine.load_checkpoint()

    model = engine.model_parts[0]
    for name, mod in model.named_modules():
        short = name.replace("._checkpoint_wrapped_module", "")
        if short.endswith(".router"):
            mod.register_forward_hook(_router_hook(short))
        elif short.endswith(".inner_experts"):
            mod.register_forward_hook(_rows_hook(short))

    pd = engine.parallel_dims
    data_iterator = trainer.microbatch_generator(trainer.dataloader)
    loss = None
    for step in range(1, steps + 1):
        ARMED["on"] = step == steps
        groups, local_valid = [], 0
        for _ in range(trainer.gradient_accumulation_steps):
            group = []
            for _ in range(trainer.num_pp_microbatches):
                mb = next(data_iterator)
                local_valid += mb.num_valid_tokens
                group.append(mb)
            groups.append(group)
        local = torch.tensor(local_valid, dtype=torch.int64, device=engine.device)
        gvt = dist_utils.dist_sum_tensor(local, pd.get_mesh("dp")) if pd.dp_enabled else local
        gvt = engine.prepare_step(gvt, num_accumulation_steps=trainer.gradient_accumulation_steps)
        acc = None
        for i, group in enumerate(groups):
            d = engine.forward_backward_microbatch(
                microbatch_group=group, global_valid_tokens=gvt, accumulation_index=i
            )
            acc = d.clone() if acc is None else acc.add_(d)
        engine.optimizer_step()
        loss = acc

    rank, size = dist.get_rank(), dist.get_world_size()
    layers = sorted({k[0] for k in CAPTURE if k[1] == "rows"})
    report = []
    for layer in layers:
        router = layer.rsplit(".", 2)[0] + ".router"
        counts = CAPTURE.get((router, "counts"))
        rows = CAPTURE[(layer, "rows")].to(torch.int64)
        E = CAPTURE[(layer, "E")]
        g_counts = counts.to(torch.int64).clone()
        dist.all_reduce(g_counts)
        maxvio = float(g_counts.max()) / (float(g_counts.sum()) / g_counts.numel()) - 1.0
        if rows.numel() > E:  # moonep: [E + B], this rank owns a slice plus its slots
            own = E // size
            mine = int(rows[rank * own:(rank + 1) * own].sum()) + int(rows[E:].sum())
        else:
            mine = int(rows.sum())
        per_rank = torch.zeros(size, dtype=torch.int64, device=rows.device)
        per_rank[rank] = mine
        dist.all_reduce(per_rank)
        v = per_rank.tolist()
        report.append({"layer": layer, "maxvio": maxvio,
                       "rank_rows": v,
                       "rank_imbalance": max(v) / (sum(v) / len(v)) if sum(v) else float("nan")})

    if rank == 0:
        mv = [r["maxvio"] for r in report]
        ri = [r["rank_imbalance"] for r in report]
        res = {"loss": float(loss.item()) if loss is not None else None,
               "steps": steps, "layers": len(report),
               "maxvio_mean": sum(mv) / len(mv), "maxvio_max": max(mv),
               "rank_imbalance_mean": sum(ri) / len(ri), "rank_imbalance_max": max(ri),
               "per_layer": report}
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"QB_PROBE_OK steps={steps} layers={len(report)} "
              f"maxvio_mean={res['maxvio_mean']:.3f} maxvio_max={res['maxvio_max']:.3f} "
              f"rank_imb_mean={res['rank_imbalance_mean']:.4f} rank_imb_max={res['rank_imbalance_max']:.4f}",
              flush=True)
    trainer.close()
    dist.barrier(); dist.destroy_process_group()


main()
