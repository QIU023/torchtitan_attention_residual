"""Local recipes (never committed) for DEP a03f74981 on 4 x H100: the B200 vision_dep cell without FSDP
(pp2 x vpp4 x tp2 x ep2), every micro-batch carrying images, with DEP off, in K2.5 mode and with the
bubble. The w_* recipes widen the debug model to DEPW_DIM (every width that is a multiple of dim
scales, heads by count; layers, blocks, experts and vocab unchanged). DEP_PARAM_DUMP saves the
initial parameters, DEP_GRAD_DUMP (with DEP_GRAD_STEPS) the gradients right before the optimizer
step, DEPN_MEM_OUT every rank's per-step peak."""

import atexit
import json
import os

import torch

from torchtitan.components.optimizer.optimizer import OptimizersContainer
from torchtitan.trainer import Trainer

_RECORDS: list[dict] = []
_inner = OptimizersContainer.step
_calls = {"n": 0}


def _full(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _dumping_step(self, closure=None):
    _calls["n"] += 1
    n, rank = _calls["n"], int(os.environ.get("RANK", "0"))
    params_out = os.environ.get("DEP_PARAM_DUMP")
    if params_out and n == 1:
        params = {
            f"{i}:{name}": _full(p.detach()).clone().cpu()
            for i, model in enumerate(self.model_parts)
            for name, p in model.named_parameters()
        }
        os.makedirs(params_out, exist_ok=True)
        torch.save(params, os.path.join(params_out, f"rank{rank}.pt"))
    grads_out = os.environ.get("DEP_GRAD_DUMP")
    steps = {int(s) for s in os.environ.get("DEP_GRAD_STEPS", "1").split(",")}
    if grads_out and n in steps:
        grads = {
            f"{i}:{name}": _full(p.grad.detach()).clone().cpu()
            for i, model in enumerate(self.model_parts)
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        os.makedirs(grads_out, exist_ok=True)
        torch.save(grads, os.path.join(grads_out, f"rank{rank}_step{n}.pt"))
    return _inner(self, closure)


OptimizersContainer.step = _dumping_step


def _record() -> None:
    _RECORDS.append(
        {
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
    )


def _install_memory() -> None:
    out = os.environ.get("DEPN_MEM_OUT")
    if not out:
        return
    from torchtitan.observability import metrics

    reset = metrics.DeviceMemoryMonitor.reset_peak_stats

    def reset_after_recording(self):
        _record()
        reset(self)

    metrics.DeviceMemoryMonitor.reset_peak_stats = reset_after_recording

    def dump() -> None:
        if not torch.cuda.is_available():
            return
        _record()
        rank = int(os.environ.get("RANK", "0"))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
            json.dump({"rank": rank, "records": _RECORDS}, f)

    atexit.register(dump)


_install_memory()


def _widened(attn_backend="flex", converters=None, moe_comm_backend="standard", *, enable_sp, seq_len=None):
    from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config

    dim = int(os.environ["DEPW_DIM"])
    s = dim // 256
    assert dim == 256 * s and converters is None
    return _kimi_k3_config(
        max_context_length=seq_len or 2048, dim=dim, enable_sp=enable_sp, moe_comm_backend=moe_comm_backend,
        vocab_size=2048, num_layers=17, full_attention_layers={3, 7, 11, 15, 16}, attn_res_block_size=4,
        num_heads=4 * s, q_lora_rank=128 * s, kv_lora_rank=64 * s, qk_nope_head_dim=64, qk_rope_head_dim=32,
        v_head_dim=64, kda_head_dim=128, conv_kernel_size=4, dense_hidden_dim=512 * s, latent_dim=128 * s,
        expert_hidden_dim=128 * s, num_experts=8, top_k=2, num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim, dim=256 * s, qkv_dim=512 * s, hidden_dim=512 * s, num_layers=2, num_heads=4 * s,
            init_pos_emb_height=32, init_pos_emb_width=32),
        attn_backend=attn_backend)


def _base(widened: bool = False) -> Trainer.Config:
    from torchtitan.models.kimi_k3 import config_registry
    from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4

    original = config_registry.model_registry
    if widened:
        config_registry.model_registry = lambda flavor, **kw: _widened(**kw)
    try:
        config = kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4()
    finally:
        config_registry.model_registry = original
    if widened:
        assert config.model.dim == int(os.environ["DEPW_DIM"])
    config.parallelism.data_parallel_shard_degree = 1
    return config


def _mode(config: Trainer.Config, enabled: bool, bubble: bool) -> Trainer.Config:
    config.model.vision_dep.enabled = enabled
    config.model.vision_dep.bubble = bubble
    return config


def dep_off() -> Trainer.Config:
    return _mode(_base(), False, False)


def dep_k25() -> Trainer.Config:
    return _mode(_base(), True, False)


def dep_bubble() -> Trainer.Config:
    return _mode(_base(), True, True)


def w_dep_off() -> Trainer.Config:
    return _mode(_base(True), False, False)


def w_dep_k25() -> Trainer.Config:
    return _mode(_base(True), True, False)


def w_dep_bubble() -> Trainer.Config:
    return _mode(_base(True), True, True)
