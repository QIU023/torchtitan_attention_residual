"""MultimodalLM = SigLIP (frozen) + MLP projector + AttnRes-Kimi-436M LM.

Structure: rather than wrapping the LM into a single nn.Module that has
SigLIP + projector as children (which would mess with FSDP wrapping
on the LM done by torchtitan's parallelize_kimi_linear), this module
keeps the components SEPARATE:

* `vision_tower` — SigLIPVisionModel, frozen, replicated on each rank
* `projector` — small MLP (Linear → GELU → Linear), trainable, replicated
* `lm` — torchtitan-built KimiLinearAttnResModel, FSDP2-sharded

Forward composes them via:
    vision_features = vision_tower(pixel_values)             # (B, N_vis, V_dim)
    vision_embeds   = projector(vision_features)             # (B, N_vis, lm_dim)
    text_embeds     = lm.embed_tokens(input_ids)             # (B, T, lm_dim)
    embeds          = scatter vision_embeds at IMAGE_TOKEN_ID positions
    h               = lm.layers(embeds)
    h               = lm.norm(h)
    logits          = lm.lm_head(h)
    loss            = CE(logits, labels, ignore=-100)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase5.multimodal_dataset import IGNORE_INDEX, IMAGE_TOKEN_ID


class Projector(nn.Module):
    """2-layer MLP from vision_dim → lm_dim."""

    def __init__(self, vision_dim: int, lm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, lm_dim, bias=True)
        self.fc2 = nn.Linear(lm_dim, lm_dim, bias=True)
        # LLaVA-1.5 init: small Gaussian, normal default works
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def lm_forward_with_embeds(lm: nn.Module, embeds: torch.Tensor) -> torch.Tensor:
    """Run the LM forward pass starting from token embeddings instead of ids.

    Bypasses lm.embed_tokens by passing the embeds directly through layers,
    norm, and lm_head. Mirrors KimiLinearModel.forward but skips the
    embedding step.
    """
    h = embeds
    # Kimi layers is a ModuleDict keyed by str(int).
    for layer in lm.layers.values():
        h = layer(h)
    if getattr(lm, "norm", None) is not None:
        h = lm.norm(h)
    if getattr(lm, "lm_head", None) is not None:
        return lm.lm_head(h)
    return h


def multimodal_loss(
    vision_tower: nn.Module,
    projector: Projector,
    lm: nn.Module,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the LM loss for a multimodal batch.

    Args:
        vision_tower: SigLIP, frozen (caller ensures no_grad context for vision).
        projector: trainable MLP.
        lm: trainable AttnRes-Kimi LM (FSDP2-wrapped).
        pixel_values: (B, 3, H, W).
        input_ids: (B, T). Image-token positions == IMAGE_TOKEN_ID.
        labels: (B, T). IGNORE_INDEX where image / BOS / pad.

    Returns:
        scalar sum-reduction CE loss on non-ignored positions.
    """
    # Vision forward — frozen, no grad. .last_hidden_state shape (B, N_vis, V_dim).
    with torch.no_grad():
        vision_out = vision_tower(pixel_values=pixel_values)
        vision_features = vision_out.last_hidden_state

    # Projector — trainable.
    vision_embeds = projector(vision_features)  # (B, N_vis, lm_dim)

    # Text embeddings via the LM's embedding table.
    # Even though IMAGE_TOKEN_ID positions get overwritten below, we still
    # need to call embed_tokens so its grad path is wired.
    text_embeds = lm.embed_tokens(input_ids)    # (B, T, lm_dim)

    # Scatter vision_embeds into text_embeds at IMAGE_TOKEN_ID positions.
    # Standard LLaVA layout: image positions are contiguous at the start
    # of each row (length N_vision), but to be defensive we scatter via
    # a boolean mask which handles any order.
    image_mask = (input_ids == IMAGE_TOKEN_ID)  # (B, T)
    expected_per_row = vision_embeds.size(1)
    n_image_per_row = image_mask.sum(dim=1)
    if not torch.all(n_image_per_row == expected_per_row):
        bad = (n_image_per_row != expected_per_row).nonzero(as_tuple=False)
        raise RuntimeError(
            f"Each row must have exactly {expected_per_row} image tokens; "
            f"row counts: min={n_image_per_row.min().item()}, "
            f"max={n_image_per_row.max().item()}, bad rows: {bad.flatten().tolist()[:5]}..."
        )
    # Replace embeddings at image positions with vision_embeds.
    # text_embeds[image_mask] is shape (B*N_vis, lm_dim); flatten vision_embeds to match.
    text_embeds = text_embeds.clone()
    text_embeds[image_mask] = vision_embeds.reshape(-1, vision_embeds.size(-1)).to(text_embeds.dtype)

    # LM forward from embeddings.
    logits = lm_forward_with_embeds(lm, text_embeds)  # (B, T, V_lm)

    # CE on non-ignored positions; sum reduction (caller normalizes by token count).
    loss = F.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )
    return loss
