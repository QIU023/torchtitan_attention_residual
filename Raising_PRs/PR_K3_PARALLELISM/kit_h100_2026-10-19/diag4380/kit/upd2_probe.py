"""#4380 diagnosis, second probe (logbook kit only): cpmm_probe's cell; every rank prints, per vision gather, how many of
its local positions take a vision-bank row and which rows (UPD2_GATHER=1, typechecking off: the checker has
no rule for the boolean index); with DUMP_DIR, rank 0 saves the full gradients before optimizer steps 1 and 2
(DUMP_DIR/grad<step>.pt)."""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import cpmm_probe


def _grads(engine, step: int) -> None:
    out = {}
    for part in engine.model_parts:
        for name, param in part.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            out[name.replace("._checkpoint_wrapped_module", "")] = g.detach().cpu()
    if dist.get_rank() == 0:
        os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
        torch.save(out, f"{os.environ['DUMP_DIR']}/grad{step}.pt")
        print(f"UPD2_DUMP grad{step} {len(out)} tensors", flush=True)


def cell():
    import torchtitan.models.kimi_k3.model as k3
    from torchtitan.training_engine import TrainingEngine

    original_gather = k3.gather_vision_embeds
    calls = [0]

    def gather(embeddings_TD, *, vision_bank_VD, vision_bank_indices_T):
        idx = vision_bank_indices_T.to_local() if isinstance(vision_bank_indices_T, DTensor) else vision_bank_indices_T
        rows = idx[idx >= 0]
        first = int(rows.min()) if rows.numel() else -1
        last = int(rows.max()) if rows.numel() else -1
        bank = vision_bank_VD.to_local() if isinstance(vision_bank_VD, DTensor) else vision_bank_VD
        print(f"UPD2_GATHER rank {dist.get_rank()} call {calls[0]} local_image_tokens {rows.numel()} rows {first}..{last} "
              f"bank_rows {bank.shape[0]}", flush=True)
        calls[0] += 1
        return original_gather(embeddings_TD, vision_bank_VD=vision_bank_VD, vision_bank_indices_T=vision_bank_indices_T)

    if os.environ.get("UPD2_GATHER") == "1":
        k3.gather_vision_embeds = gather
    original_step = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps < 2 and os.environ.get("DUMP_DIR"):
            _grads(self, self.num_completed_steps + 1)
        return original_step(self)

    TrainingEngine.optim_step = optim_step
    print(f"[upd2_probe] rank {os.environ.get('RANK')} patched", flush=True)
    return cpmm_probe.cell()
