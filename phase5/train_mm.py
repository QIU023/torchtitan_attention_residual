"""Multimodal full-parameter fine-tune of AttnRes-Kimi-436M.

Subclasses torchtitan's Trainer to reuse all the FSDP / optim /
scheduler / checkpoint plumbing. The override:

* swap the dataloader for `LlavaPretrainDataset`
* attach a frozen SigLIP vision tower + a trainable MLP projector
  to the trainer
* extend the optimizer to also step the projector params
* override `forward_backward_step` to do the multimodal forward
  (vision → projector → embed scatter → LM → CE)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

WORKSPACE = Path(__file__).resolve().parent.parent
TORCHTITAN_PATH = WORKSPACE / "torchtitan"
for p in (str(WORKSPACE), str(TORCHTITAN_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torchtitan.trainer import Trainer  # noqa: E402
from torchtitan.tools.logging import init_logger, logger  # noqa: E402
from torchtitan.components.dataloader import ParallelAwareDataloader  # noqa: E402
import torchtitan.distributed.utils as dist_utils  # noqa: E402

from phase5.multimodal_dataset import (  # noqa: E402
    IGNORE_INDEX, LlavaPretrainDataset, collate_with_pad,
)
from phase5.multimodal_model import Projector, multimodal_loss  # noqa: E402


# ----------------------------------------------------------------------
# CLI args (consumed before torchtitan's tyro)
# ----------------------------------------------------------------------


def _parse_mm_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mm.json", dest="mm_json", required=True,
                   help="LLaVA-Pretrain caption json path")
    p.add_argument("--mm.images", dest="mm_images", required=True,
                   help="LLaVA-Pretrain images dir")
    p.add_argument("--mm.vision-model", dest="mm_vision_model",
                   default="google/siglip-base-patch16-224")
    p.add_argument("--mm.tokenizer", dest="mm_tokenizer",
                   default="NousResearch/Meta-Llama-3.1-8B",
                   help="Tokenizer source (must match the LM ckpt's pretrain tokenizer)")
    p.add_argument("--mm.cache-dir", dest="mm_cache_dir",
                   default="/root/hf_cache")
    p.add_argument("--mm.proj-lr-mult", dest="mm_proj_lr_mult",
                   type=float, default=1.0,
                   help="LR multiplier for projector params relative to LM LR")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


# ----------------------------------------------------------------------
# Trainer subclass
# ----------------------------------------------------------------------


class MultimodalTrainer(Trainer):
    def __init__(self, config, *,
                 json_path: str, images_dir: str, vision_model: str,
                 tokenizer_path: str, cache_dir: str,
                 proj_lr_mult: float = 1.0):
        super().__init__(config)

        # ----- DP rank info -----
        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            dp_world_size = batch_mesh.size()
            dp_rank = batch_mesh.get_local_rank()
        else:
            dp_world_size, dp_rank = 1, 0

        # ----- Vision tower (frozen, replicated) -----
        logger.info(f"mm: loading vision_tower {vision_model}")
        vision = AutoModel.from_pretrained(
            vision_model, torch_dtype=torch.bfloat16,
            cache_dir=cache_dir, low_cpu_mem_usage=True,
        )
        # SigLIP has both vision_model and text_model; we only want the vision side.
        if hasattr(vision, "vision_model"):
            self.vision_tower = vision.vision_model.to(self.device).eval()
        else:
            self.vision_tower = vision.to(self.device).eval()
        for p in self.vision_tower.parameters():
            p.requires_grad_(False)
        # Vision token count (inferred): SigLIP-Base patch16 224x224 → 196.
        # Vision feature dim: SigLIP-Base = 768, SO400M = 1152.
        # We'll pull lm_dim from the model_config below.
        vision_dim = getattr(vision.config, "vision_config", vision.config).hidden_size
        logger.info(f"mm: vision_tower hidden_size={vision_dim}, frozen")

        # ----- Image processor + tokenizer -----
        self.image_processor = AutoProcessor.from_pretrained(
            vision_model, cache_dir=cache_dir,
        )
        self.mm_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, cache_dir=cache_dir,
        )

        # ----- LM dim (Kimi 436M = 1168) -----
        lm = self.model_parts[0]
        # KimiLinearModel exposes embed_tokens with weight shape (V, D).
        lm_dim = lm.embed_tokens.weight.shape[1] if hasattr(lm, "embed_tokens") and lm.embed_tokens is not None else None
        if lm_dim is None:
            raise RuntimeError("Cannot infer lm_dim from model_parts[0].embed_tokens")
        logger.info(f"mm: lm_dim={lm_dim}")

        # ----- Projector (trainable, replicated) -----
        self.projector = Projector(vision_dim=vision_dim, lm_dim=lm_dim).to(
            device=self.device, dtype=torch.bfloat16,
        )
        n_proj_params = sum(p.numel() for p in self.projector.parameters())
        logger.info(f"mm: projector built, params={n_proj_params:,}")

        # ----- Build a separate optimizer for the projector -----
        # We can't add_param_group to the LM's optimizer because
        # torchtitan's LambdaLR was built for the original param groups
        # only and asserts strict zip(groups, lr_values) — adding a
        # group breaks it. A fresh AdamW for the projector, appended
        # to OptimizersContainer.optimizers, is stepped by the same
        # optimizers.step() / zero_grad() loop without needing a
        # scheduler (projector keeps a fixed LR).
        proj_lr = config.optimizer.lr * proj_lr_mult
        proj_optim = torch.optim.AdamW(
            list(self.projector.parameters()),
            lr=proj_lr,
            betas=(0.9, 0.95), weight_decay=0.01,
        )
        self.optimizers.optimizers.append(proj_optim)
        logger.info(f"mm: appended projector AdamW (lr={proj_lr}, fixed) to "
                    f"OptimizersContainer; total inner optimizers="
                    f"{len(self.optimizers.optimizers)}")

        # ----- Replace dataloader -----
        ds = LlavaPretrainDataset(
            json_path=json_path,
            images_dir=images_dir,
            tokenizer=self.mm_tokenizer,
            image_processor=self.image_processor,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        pad_id = self.mm_tokenizer.pad_token_id or 0
        self.dataloader = ParallelAwareDataloader(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=config.training.local_batch_size,
            collate_fn=lambda b: collate_with_pad(b, pad_id=pad_id),
            num_workers=0,   # > 0 would corrupt CUDA context; see phase5_distillation_deprecated/miniplm
        )
        logger.info(
            f"mm: replaced dataloader with LlavaPretrainDataset "
            f"(N={len(ds.records):,}, dp_rank={dp_rank}/{dp_world_size}, "
            f"local_bs={config.training.local_batch_size})"
        )

    # ------------------------------------------------------------------
    # The single override.
    # ------------------------------------------------------------------
    def forward_backward_step(self, *, input_dict, labels,
                              global_valid_tokens) -> torch.Tensor:
        if self.parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Multimodal trainer does not support PP. Run on FSDP only."
            )

        pixel_values = input_dict["pixel_values"].to(
            self.device, dtype=torch.bfloat16, non_blocking=True,
        )
        input_ids = input_dict["input"].to(self.device, non_blocking=True)
        labels_ = labels.to(self.device, non_blocking=True)

        with self.train_context():
            loss_sum = multimodal_loss(
                vision_tower=self.vision_tower,
                projector=self.projector,
                lm=self.model_parts[0],
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels_,
            )
            # global_valid_tokens is the count of non-IGNORE labels across
            # the DP mesh — train_step pre-computes this.
            loss = loss_sum / global_valid_tokens
            loss.backward()
        return loss


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main():
    init_logger()
    mm_args = _parse_mm_args()

    from torchtitan.config import ConfigManager
    cm = ConfigManager()
    config = cm.parse_args(sys.argv[1:])

    # post_dataloading_process expects input_dict["input"] to be the main
    # token tensor; for multimodal we replaced the dataloader and override
    # forward_backward_step. We need to also override post_dataloading_process
    # because train_step calls it via forward_backward_step.
    # Easier: train_step will call our overridden forward_backward_step,
    # and our override does NOT call post_dataloading_process — it consumes
    # the multimodal batch directly. So no additional override needed.

    # However train_step counts local_valid_tokens via labels != IGNORE_INDEX
    # which happens to be the right count for us (image positions are
    # IGNORE_INDEX, so they don't contribute, perfect).

    trainer = MultimodalTrainer(
        config,
        json_path=mm_args.mm_json,
        images_dir=mm_args.mm_images,
        vision_model=mm_args.mm_vision_model,
        tokenizer_path=mm_args.mm_tokenizer,
        cache_dir=mm_args.mm_cache_dir,
        proj_lr_mult=mm_args.mm_proj_lr_mult,
    )
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
