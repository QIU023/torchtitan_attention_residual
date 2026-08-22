"""Give some ranks images and others none, and check nothing deadlocks.

4272's P1 asks for mixed image/text rank batches under FSDP. The model handles
it deliberately -- multimodal_model gates the sentinel exchange on the mesh
rather than on the data, and runs the tower on a placeholder when a rank's batch
has no images -- so both the CP collective and FSDP's tower all-gather are still
issued. Neither path had a distributed test: test_text_only_path is single
process, so it exercises the branch but not the collectives it exists for.

The odd ranks drop pixel_values here. A failure is a hang, not an exception, so
this runs under a timeout and the absence of output is the signal.

Usage (4 GPUs):
  torchrun --nproc_per_node=4 mixed_rank_image_harness.py <train args...>
"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    if "--" in sys.argv:
        sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1 :]
    import torchtitan.trainer as trainer_mod
    from torchtitan.train import main as train_main

    original = trainer_mod.Trainer.train_step

    def patched(self, *args, **kwargs):
        if not getattr(self, "_mr_installed", False):
            self._mr_installed = True
            rank = torch.distributed.get_rank()
            drop = rank % 2 == 1
            for part in self.model_parts:
                if type(part).__name__ != "KimiK3MultimodalModel":
                    continue
                inner = part.forward

                def fwd(*a, _inner=inner, _drop=drop, **kw):
                    if _drop:
                        kw["pixel_values"] = None
                        kw.pop("grid_thw", None)
                        kw.pop("image_grid_hws", None)
                    return _inner(*a, **kw)

                part.forward = fwd
            print(f"[MR] rank {rank} drops images: {drop}", flush=True)
        return original(self, *args, **kwargs)

    trainer_mod.Trainer.train_step = patched
    train_main()


if __name__ == "__main__":
    main()
