"""Logbook probe: train the cell, then log every LoRA module whose merged weight differs from the gathered expectation,
with its type, dtypes, placements and the value at the worst element, and the same expectation computed in the
local batch of experts (ENTRYPOINT=this)."""
import logging
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.config import ConfigLoader
from torchtitan.models.common.lora import _LoRAGroupedLinearMixin, _LoRAMixin, _state_dict_prefix, merge_lora_state_dict
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)


def _full(t):
    return t.full_tensor() if isinstance(t, DTensor) else t


def _desc(t):
    return f"{type(t).__name__}{tuple(t.shape)} {t.dtype} {t.placements if isinstance(t, DTensor) else ''}"


def main():
    init_logger()
    config = cast(Trainer.Config, ConfigLoader().load())
    trainer = config.build()
    rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        trainer.train()
        for part in trainer.engine.model_parts:
            state = part.state_dict()
            merged = merge_lora_state_dict(part)
            with torch.no_grad():
                for name, module in part.named_modules():
                    if not isinstance(module, _LoRAMixin):
                        continue
                    prefix = _state_dict_prefix(name, state)
                    key = f"{prefix}.weight" if prefix else "weight"
                    base = _full(state[key])
                    a, b = _full(module.lora_a.weight).float(), _full(module.lora_b.weight).float()
                    grouped = isinstance(module, _LoRAGroupedLinearMixin)
                    delta = torch.bmm(b.flatten(1, -2), a) if grouped else torch.matmul(b, a)
                    expected = (base.float() + module._lora_scaling * delta.view(base.shape)).to(base.dtype)
                    got = _full(merged[key])
                    diff = (got.float() - expected.float()).abs()
                    if diff.max().item() == 0:
                        continue
                    idx = tuple(int(i) for i in torch.unravel_index(diff.argmax(), diff.shape))
                    chunked = ""
                    if grouped and isinstance(module.lora_a.weight, DTensor):
                        n_local = module.lora_a.weight.to_local().shape[0]
                        parts = [torch.bmm(b[i:i + n_local].flatten(1, -2), a[i:i + n_local]) for i in range(0, a.shape[0], n_local)]
                        delta_c = torch.cat(parts).view(base.shape)
                        exp_c = (base.float() + module._lora_scaling * delta_c).to(base.dtype)
                        chunked = f" chunked({n_local}) max diff {(got.float() - exp_c.float()).abs().max().item():.3e}"
                    if rank == 0:
                        logger.info(
                            "LOCATE %s %s grouped=%s base %s | a %s | b %s | max diff %.3e at %s: got %.9g expected %.9g base %.9g%s",
                            name, type(module).__name__, grouped, _desc(state[key]), _desc(module.lora_a.weight),
                            _desc(module.lora_b.weight), diff.max().item(), idx, got[idx].item(), expected[idx].item(),
                            base[idx].item(), chunked)
        logger.info("LOCATE_DONE rank %d", rank)
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
