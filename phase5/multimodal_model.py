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


def lm_forward_multimodal(lm: nn.Module, input_ids: torch.Tensor,
                          vision_embeds: torch.Tensor,
                          image_mask: torch.Tensor) -> torch.Tensor:
    """Run the LM with vision-token embedding scatter happening INSIDE
    the LM forward (single FSDP-root call).

    KimiLinearModel.forward / KimiLinearAttnResModel.forward both
    accept the (vision_embeds, image_mask) kwargs after the multimodal
    patch — they call embed_tokens internally and overwrite the
    image-mask positions with vision_embeds before running layers.
    """
    return lm(input_ids, vision_embeds=vision_embeds, image_mask=image_mask)


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

    # Build the image-token mask. Variable image count per row supported:
    # row i may have any 0 ≤ n_i ≤ vision_embeds.size(1) image tokens.
    # The LM's forward filter (attn_res_model.py) consumes only the leading
    # n_i slots of vision_embeds[i] per row, so callers can pad
    # vision_embeds to a fixed width (PP shape stability) without forcing
    # all rows to have the same image count.
    image_mask = (input_ids == IMAGE_TOKEN_ID)  # (B, T)
    n_image_per_row = image_mask.sum(dim=1)
    n_vis_max = vision_embeds.size(1)
    if (n_image_per_row > n_vis_max).any():
        bad = (n_image_per_row > n_vis_max).nonzero(as_tuple=False)
        raise RuntimeError(
            f"Row image count exceeds vision_embeds slots ({n_vis_max}); "
            f"row counts max={n_image_per_row.max().item()}, "
            f"bad rows: {bad.flatten().tolist()[:5]}..."
        )

    # Single FSDP-root LM call: embed_tokens + image-position scatter +
    # layers + norm + lm_head all happen inside lm.forward.
    logits = lm_forward_multimodal(
        lm, input_ids=input_ids, vision_embeds=vision_embeds, image_mask=image_mask,
    )  # (B, T, V_lm)

    # CE on non-ignored positions; sum reduction (caller normalizes by token count).
    loss = F.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )
    return loss
