"""Entry point for the LoRA matrix cells, logbook kit only: torchtitan.train, then the LoRA export functions on every
model part of the trained model, under the cell's parallelism. Each rank checks that merge_lora_state_dict returns the
state dict's keys without the adapter keys, that trainable_state_dict is exactly the adapters, and that every merged
weight equals base + (alpha / rank) * B @ A on the gathered tensors (grouped products in the merge's local expert batch), then logs one EXPORT_CHECK line (ENTRYPOINT=this)."""

import logging
import os
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.config import ConfigLoader
from torchtitan.models.common.lora import (
    _LoRAGroupedLinearMixin,
    _LoRAMixin,
    _state_dict_prefix,
    merge_lora_state_dict,
    trainable_state_dict,
)
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _check(part: torch.nn.Module) -> tuple[int, float]:
    state = part.state_dict()
    merged = merge_lora_state_dict(part)
    adapters = {key for key in state if ".lora_a." in key or ".lora_b." in key}
    assert set(merged) == set(state) - adapters, "merged keys differ from the base keys"
    assert set(trainable_state_dict(part)) == adapters, "trainable parameters are not the adapters"
    count, worst = 0, 0.0
    with torch.no_grad():
        for name, module in part.named_modules():
            if not isinstance(module, _LoRAMixin):
                continue
            prefix = _state_dict_prefix(name, state)
            key = f"{prefix}.weight" if prefix else "weight"
            base = _full(state[key])
            lora_a, lora_b = _full(module.lora_a.weight).float(), _full(module.lora_b.weight).float()
            if isinstance(module, _LoRAGroupedLinearMixin):
                # cuBLAS bmm results vary with the batch count on sm120, so the expectation uses the merge's local
                # expert batch; a placement error still shows, a rounding difference between batch sizes does not.
                local = module.lora_a.weight
                n = local.to_local().shape[0] if isinstance(local, DTensor) else lora_a.shape[0]
                delta = torch.cat([torch.bmm(lora_b[i:i + n].flatten(1, -2), lora_a[i:i + n]) for i in range(0, lora_a.shape[0], n)])
            else:
                delta = torch.matmul(lora_b, lora_a)
            expected = (base.float() + module._lora_scaling * delta.view(base.shape)).to(base.dtype)
            worst = max(worst, (_full(merged[key]).float() - expected.float()).abs().max().item())
            count += 1
    return count, worst


def main() -> None:
    init_logger()
    config = cast(Trainer.Config, ConfigLoader().load())
    trainer = config.build()
    try:
        if config.create_seed_checkpoint:
            trainer.engine.save_checkpoint(last_step=True)
        else:
            trainer.train()
            total, worst = 0, 0.0
            for part in trainer.engine.model_parts:
                count, part_worst = _check(part)
                total, worst = total + count, max(worst, part_worst)
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info("EXPORT_CHECK rank %d: %d LoRA modules merged, worst |merged - expected| %.3e", rank, total, worst)
            assert worst == 0.0, f"merged weights differ from base + delta by {worst}"
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
