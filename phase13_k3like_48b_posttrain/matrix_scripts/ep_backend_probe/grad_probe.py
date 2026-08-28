"""One microbatch forward+backward from the seed checkpoint; dump per-parameter grad norms."""
import json, os, sys
import torch, torch.distributed as dist
from torch.distributed.tensor import DTensor
from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils

def main():
    init_logger()
    out_path = os.environ["GRAD_PROBE_OUT"]
    config = ConfigManager().parse_args()
    sl.init_structured_logger(source="training", output_dir=config.dump_folder, enable=False)
    trainer = config.build()
    trainer.checkpointer.load(step=config.checkpoint.load_step)
    it = trainer.batch_generator(trainer.dataloader)
    input_dict, labels = next(it)
    for k, v in list(input_dict.items()):
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(trainer.device)
    labels = labels.to(trainer.device)
    local_valid = (labels != IGNORE_INDEX).sum().to(trainer.device)
    pd = trainer.parallel_dims
    gvt = dist_utils.dist_sum_tensor(local_valid, pd.get_mesh("batch")) if pd.dp_enabled else local_valid
    trainer.optimizers.zero_grad(set_to_none=True)
    loss = trainer.forward_backward_step(input_dict=input_dict, labels=labels, global_valid_tokens=gvt)
    names, vals = [], []
    for name, p in trainer.model_parts[0].named_parameters():
        g = p.grad
        if g is None:
            v = torch.zeros((), device=trainer.device)
        else:
            if isinstance(g, DTensor):
                g = g.to_local()
            elif dist.get_rank() != 0:
                g = torch.zeros((), device=trainer.device)  # replicated: count once
            v = g.detach().float().pow(2).sum()
        names.append(name); vals.append(v)
    t = torch.stack(vals)
    dist.all_reduce(t)
    lv = float(loss.full_tensor() if isinstance(loss, DTensor) else loss)
    if dist.get_rank() == 0:
        res = {"loss": lv, "grads": {n: float(v) ** 0.5 for n, v in zip(names, t.tolist())}}
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"GRAD_PROBE_OK {out_path} loss={lv:.6f} n_params={len(names)} total_norm={float(t.sum())**0.5:.5f}", flush=True)
    trainer.close()
    dist.barrier(); dist.destroy_process_group()

main()
