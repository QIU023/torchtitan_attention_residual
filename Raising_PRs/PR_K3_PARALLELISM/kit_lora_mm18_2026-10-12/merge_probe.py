"""Logbook probe: build the cell's trainer, then report the placements of every LoRA module whose merge raises."""
import collections, logging, re
from typing import cast
import torch, torch.distributed as dist
from torch.distributed.tensor import DTensor
from torchtitan.config import ConfigLoader
from torchtitan.models.common.lora import _LoRAMixin, _adapter_delta
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)


def desc(t):
    if isinstance(t, DTensor):
        return f"DTensor{tuple(t.shape)} local{tuple(t.to_local().shape)} {t.device_mesh.mesh_dim_names} {t.placements}"
    return f"Tensor{tuple(t.shape)}"


def main():
    init_logger()
    config = cast(Trainer.Config, ConfigLoader().load())
    trainer = config.build()
    try:
        seen = collections.Counter()
        for part in trainer.engine.model_parts:
            for name, module in part.named_modules():
                if not isinstance(module, _LoRAMixin):
                    continue
                key = re.sub(r"\.\d+\.", ".N.", name)
                try:
                    with torch.no_grad():
                        _adapter_delta(module)
                    seen[(key, "ok")] += 1
                except Exception as e:
                    if seen[(key, "fail")] == 0:
                        logger.info("MERGE_FAIL %s %s: %s | base %s | a %s | b %s", key, type(module).__name__, str(e)[:100],
                                    desc(module.weight), desc(module.lora_a.weight), desc(module.lora_b.weight))
                    seen[(key, "fail")] += 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info("MERGE_SUMMARY rank %d %s", rank, sorted(seen.items()))
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
