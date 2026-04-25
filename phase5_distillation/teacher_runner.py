# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Teacher model wrapper for KD.

Loads a HuggingFace causal LM (default: Kimi-Linear-48B-A3B-Base) in
bf16, FSDP2-shards each transformer layer across the data-parallel
mesh, runs forward in eval / no_grad. Returns full-vocab logits per
rank (each rank handles its own data shard).

Memory budget on 4× RTX 5090 (31 GiB / rank):
* Teacher params bf16  : 96 GB total → 24 GB / rank after FSDP shard
* Per-step gather temp : ~1 layer's full params (~0.8 GB for 48B/120L)
* Activations          : tiny (B=1-2, T=2048, no KV cache for KD)
* Total                : ~26 GB / rank — fits with student + KD
                          intermediates on top.

NOT vLLM. vLLM exposes top-K logprobs only; KL needs full-vocab
softmax. Use plain ``transformers`` with ``trust_remote_code=True`` so
Kimi's custom modeling files (KDA + MLA + MoE) are loaded.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


# Default: Llama-3.1-8B-Base. Matches the Llama-3.1 tokenizer that
# torchtitan's default kimi_linear configs ship with
# (./assets/hf/Llama-3.1-8B). Student forward + teacher forward both
# consume Llama BPE tokens (vocab 128,256).
#
# Using NousResearch's redistribution (same weights as
# meta-llama/Llama-3.1-8B but non-gated, so HF auth is not required).
# Switch to Llama-3.1-70B-Base (int8 / 4bit) for stronger signal
# once the 8B path is validated.
DEFAULT_TEACHER = "NousResearch/Meta-Llama-3.1-8B"


class TeacherRunner:
    """Wrap a HF causal-LM in FSDP2 + eval / no_grad for KD logit production.

    Construction is via :meth:`load` (handles HF download +
    FSDP wrap). The instance is callable: ``runner(input_ids)`` returns
    ``[B, T, V]`` logits without building any autograd graph.

    The wrapper does NOT expose generate() / sampling / KV cache —
    those are unused in KD. It only exposes the single full-sequence
    forward.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def load(
        cls,
        model_name_or_path: str = DEFAULT_TEACHER,
        *,
        device_mesh: "DeviceMesh",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | None = None,
    ) -> "TeacherRunner":
        """Download (if needed), build, and FSDP2-shard the teacher.

        Args:
            model_name_or_path: HF repo id or local path.
            device_mesh: 1-D DP mesh; must have the same world size as
                the student's DP mesh so each rank's data shard can
                go through both models.
            dtype: weight + activation dtype. bf16 default; pass
                ``torch.float16`` if running on a card without bf16
                tensor cores (5090 has bf16, so default is fine).
            cache_dir: forwarded to ``transformers.AutoModelForCausalLM``;
                use this to point at an existing local snapshot if you
                pre-downloaded with ``huggingface-cli``.
        """
        from transformers import AutoModelForCausalLM
        from torch.distributed._composable.fsdp import (
            MixedPrecisionPolicy,
            fully_shard,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

        # Freeze before sharding. FSDP2 still all-gathers for forward
        # but skips reduce-scatter on backward when no grad.
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype, reduce_dtype=dtype, output_dtype=dtype,
        )

        # Locate the per-layer ModuleList. HF Kimi-Linear uses
        # ``model.model.layers``; Qwen / Llama variants use the same
        # path. Fail loudly if the layout is unexpected.
        decoder = getattr(model, "model", None)
        layers = getattr(decoder, "layers", None) if decoder is not None else None
        if layers is None:
            raise RuntimeError(
                f"TeacherRunner: cannot find ``model.model.layers`` on "
                f"{type(model).__name__}. Add an explicit shim if a "
                f"future HF backbone changes layout."
            )

        for layer in layers:
            fully_shard(layer, mesh=device_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=device_mesh, mp_policy=mp_policy)
        model.eval()

        return cls(model)

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns ``[B, T, V]`` logits.

        Caller already ensures ``input_ids`` is on the local device
        and matches the student's batch shard for this rank. We don't
        do attention masks — Kimi Linear's KDA + MLA paths handle
        causal masking internally; padding-aware KD requires extra
        plumbing not implemented here.
        """
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits
