"""Scratch flavor for the PP block-memory measurement (never committed).

The debug model's layout at a wider hidden size: PPMEM_LAYERS layers in blocks of
PPMEM_BLOCK (32 in blocks of 4 gives the released model's 8 blocks), full attention
every fourth layer, small MoE and FFN so the blocks dominate, AdamW (DistMuon refuses a
stage without matrices), full activation checkpointing so a layer keeps only its inputs.
PPMEM_SEQ tokens per micro-batch. titan resets the peak statistics after every logged
step, so the probe records them just before each reset: one record per step, with the
bytes held by the stages' receive buffers at that point. Every rank writes its records
to $PPMEM_OUT/rank<r>.json at exit.
"""

import atexit
import json
import os

import torch

from torchtitan.trainer import Trainer


_RECORDS: list[dict] = []
_STAGES: list = []


def _recv_buffer_bytes() -> int:
    seen, total = set(), 0
    for stage in _STAGES:
        infos = [i for chunk in stage.args_recv_info.values() for i in chunk]
        infos += [i for chunk in stage.grad_recv_info.values() for i in (chunk or ())]
        for info in infos:
            buf = getattr(info, "buffer", None)
            if isinstance(buf, torch.Tensor) and buf.is_cuda:
                key = buf.untyped_storage().data_ptr()
                if key not in seen:
                    seen.add(key)
                    total += buf.untyped_storage().nbytes()
    return total


def _record_peak() -> None:
    stats = torch.cuda.memory_stats()
    _RECORDS.append(
        {
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "allocated_now_gib": torch.cuda.memory_allocated() / 2**30,
            "recv_buffers_gib": _recv_buffer_bytes() / 2**30,
            "alloc_retries": stats.get("num_alloc_retries", 0),
            "num_ooms": stats.get("num_ooms", 0),
        }
    )


def _install_hooks() -> None:
    from torch.distributed.pipelining import PipelineStage

    from torchtitan.observability import metrics

    reset = metrics.DeviceMemoryMonitor.reset_peak_stats

    def reset_after_recording(self):
        _record_peak()
        reset(self)

    metrics.DeviceMemoryMonitor.reset_peak_stats = reset_after_recording
    init = PipelineStage.__init__

    def init_and_register(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _STAGES.append(self)

    PipelineStage.__init__ = init_and_register


def _dump_peak_memory() -> None:
    out = os.environ.get("PPMEM_OUT")
    if not out or not torch.cuda.is_available():
        return
    _record_peak()
    rank = int(os.environ.get("RANK", "0"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
        json.dump({"rank": rank, "records": _RECORDS}, f)


def ppmem_wide() -> Trainer.Config:
    from torchtitan.components.optimizer.optimizer import default_adamw
    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    dim = int(os.environ.get("PPMEM_DIM", "2048"))
    num_layers = int(os.environ.get("PPMEM_LAYERS", "32"))
    block = int(os.environ.get("PPMEM_BLOCK", "4"))
    seq = int(os.environ.get("PPMEM_SEQ", "4096"))
    config = kimi_k3_debugmodel(seq_len=seq)
    config.optimizer = default_adamw(lr=8e-4)
    config.activation_checkpoint = FullAC.Config()
    config.model = _kimi_k3_config(
        max_context_length=seq,
        dim=dim,
        vocab_size=2048,
        num_layers=num_layers,
        full_attention_layers={i for i in range(3, num_layers, 4)},
        attn_res_block_size=block,
        num_heads=16,
        q_lora_rank=512,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=2 * dim,
        latent_dim=256,
        expert_hidden_dim=256,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=256,
            qkv_dim=512,
            hidden_dim=512,
            num_layers=2,
            num_heads=4,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend="flex",
    )
    _install_hooks()
    atexit.register(_dump_peak_memory)
    return config
