"""Local recipes and probe (never committed): the Kimi K3 debug model (dim 256, 17 layers) with LOAD_E routed
experts, top-LOAD_K and 2 shared experts, at FSDP4 x EP4 and seq 512, AdamW(lr=8e-4), on the standard or the
MoonEP backend.

- LOAD_SKEW > 0 adds a fixed Zipf-like bias, -LOAD_SKEW * log(1 + e), to the router scores used to choose
  experts (not to the gating values), so low expert ids, homed on EP rank 0, are hot. 0 keeps natural routing.
- LOAD_NO_BIAS=1 turns the expert-bias updates off (K3's quantile balancing hook and any load_balance_coeff), so
  the skew is not balanced away during the run.
- The probe records, for every MoE layer and micro-batch, the tokens each EP rank would receive with static
  expert placement: the router's per-expert counts summed over the EP group, then over each rank's home
  experts. On MoonEP it also checks, per dispatch, that this rank's real rows (the padded group ends minus
  the padding) equal S x K, and counts the slots holding a copied expert.
- Writes $LOAD_OUT/rank<r>.json at exit.
"""

import atexit
import json
import os

import torch
import torch.distributed as dist

from torchtitan.trainer import Trainer

_E = int(os.environ.get("LOAD_E", "128"))
_K = int(os.environ.get("LOAD_K", "8"))
_SKEW = float(os.environ.get("LOAD_SKEW", "0"))
_CALLS: list[dict] = []
_STEPS: list[dict] = []
_LAYERS: dict[int, int] = {}
_GROUP: list = [None]
_INSTALLED: list = [False]


def _in_backward() -> bool:
    return torch._C._current_graph_task_id() != -1


def _record_call(module, x_TD, counts_E) -> None:
    if not torch.is_grad_enabled() or _in_backward():
        return
    layer = _LAYERS.setdefault(id(module), len(_LAYERS))
    _CALLS.append({"layer": layer, "tokens": int(x_TD.shape[0]), "counts": counts_E.detach().to(torch.int64).clone()})
    if _GROUP[0] is None and getattr(module.token_dispatcher, "ep_mesh", None) is not None:
        _GROUP[0] = module.token_dispatcher.ep_mesh.get_group()


def _moonep_rows(metadata, tokens: int) -> dict:
    cu = metadata.cu_seqlens
    padded = int(cu[-1].item()) if cu is not None and cu.numel() else -1
    plan = metadata.plan
    padding = -1
    ranges = getattr(plan, "zero_fill_ranges", None)
    if isinstance(ranges, torch.Tensor):
        padding = int((ranges[:, 1] - ranges[:, 0]).clamp(min=0).sum().item())
    rank = dist.get_rank(_GROUP[0]) if _GROUP[0] is not None else 0
    copies = getattr(plan, "experts_to_copy", None)
    slots = int((copies[rank] >= 0).sum().item()) if isinstance(copies, torch.Tensor) else -1
    return {
        "padded_rows": padded,
        "padding_rows": padding,
        "real_rows": padded - padding if padding >= 0 else -1,
        "expected_rows": tokens * _K,
        "slots_used": slots,
    }


def _summarize_step() -> None:
    if not _CALLS:
        return
    counts = torch.stack([c["counts"] for c in _CALLS])
    group = _GROUP[0]
    size = 1
    if group is not None:
        dist.all_reduce(counts, group=group)
        size = dist.get_world_size(group)
    per_rank = counts.view(len(_CALLS), size, -1).sum(-1).float()
    ratio = (per_rank.max(dim=1).values / per_rank.mean(dim=1).clamp(min=1)).tolist()
    by_layer: dict[int, list[float]] = {}
    for call, r in zip(_CALLS, ratio, strict=True):
        by_layer.setdefault(call["layer"], []).append(r)
    step = {
        "max_over_mean_by_layer": {str(k): max(v) for k, v in sorted(by_layer.items())},
        "mean_max_over_mean": sum(ratio) / len(ratio),
        "worst_max_over_mean": max(ratio),
        "calls": len(_CALLS),
        "moonep": [c["moonep"] for c in _CALLS if "moonep" in c],
    }
    _STEPS.append(step)
    _CALLS.clear()


def _install() -> None:
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True
    from torchtitan.components.optimizer.optimizer import OptimizersContainer
    from torchtitan.models.common import moe, token_dispatcher

    classes = [moe.RoutedExperts]
    if hasattr(moe, "MoonEPRoutedExperts"):
        classes.append(moe.MoonEPRoutedExperts)
    def recording(original):
        # Same positional names as RoutedExperts.forward: local_spmd maps inputs to layouts by name.
        def forward(self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E):
            _record_call(self, x_TD, num_local_tokens_per_expert_E)
            return original(self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E)

        return forward

    for cls in classes:
        cls.forward = recording(cls.forward)

    if hasattr(token_dispatcher, "MoonEPTokenDispatcher"):
        dispatch = token_dispatcher.MoonEPTokenDispatcher.dispatch

        def moonep_dispatch(self, x_TD, *args, _dispatch=dispatch):
            out = _dispatch(self, x_TD, *args)
            if _CALLS and torch.is_grad_enabled() and not _in_backward():
                _CALLS[-1]["moonep"] = _moonep_rows(out[2], int(x_TD.shape[0]))
            return out

        token_dispatcher.MoonEPTokenDispatcher.dispatch = moonep_dispatch

    if _SKEW:
        bias_E = -_SKEW * torch.log1p(torch.arange(_E, dtype=torch.float32))
        # K3 routes with QuantileBalancedTopKRouter, whose training path picks experts itself.
        routers = [getattr(moe, "QuantileBalancedTopKRouter", moe.TokenChoiceTopKRouter)]
        for cls in routers:
            select = cls.__dict__["_select_experts"]

            def skewed_select(self, scores_TE, expert_bias_E=None, *args, _select=select, **kwargs):
                b = bias_E.to(scores_TE.device)
                biased = b if expert_bias_E is None else expert_bias_E + b
                return _select(self, scores_TE, biased, *args, **kwargs)

            cls._select_experts = skewed_select

    if os.environ.get("LOAD_NO_BIAS") == "1":
        moe.register_moe_quantile_balancing_hook = lambda *args, **kwargs: None

    step = OptimizersContainer.step

    def summarizing_step(self, *args, _step=step, **kwargs):
        _summarize_step()
        return _step(self, *args, **kwargs)

    OptimizersContainer.step = summarizing_step

    def dump() -> None:
        out = os.environ.get("LOAD_OUT")
        if not out:
            return
        rank = int(os.environ.get("RANK", "0"))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
            json.dump({"rank": rank, "E": _E, "K": _K, "skew": _SKEW, "steps": _STEPS}, f)

    atexit.register(dump)


def _without_bias_update(obj, seen=None) -> None:
    import dataclasses

    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if hasattr(obj, "load_balance_coeff"):
            obj.load_balance_coeff = None
        for f in dataclasses.fields(obj):
            _without_bias_update(getattr(obj, f.name), seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _without_bias_update(v, seen)
    elif isinstance(obj, dict):
        for v in obj.values():
            _without_bias_update(v, seen)


def _model(backend: str):
    from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config

    dim = 256
    return _kimi_k3_config(
        max_context_length=512,
        dim=dim,
        enable_sp=True,
        moe_comm_backend=backend,
        vocab_size=2048,
        num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16},
        attn_res_block_size=4,
        num_heads=4,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=512,
        latent_dim=128,
        expert_hidden_dim=128,
        num_experts=_E,
        top_k=_K,
        num_shared_experts=2,
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


def _cell(backend: str) -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(seq_len=512)
    config.model = _model(backend)
    if os.environ.get("LOAD_NO_BIAS") == "1":
        _without_bias_update(config.model)
    try:
        from torchtitan.components.optimizer import AdamW, OptimizersContainer

        config.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    except ImportError:
        from torchtitan.components.optimizer import default_adamw

        config.optimizer = default_adamw(lr=8e-4)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 4
    mode = os.environ.get("MOONEP_AC")
    if mode == "none":
        config.activation_checkpoint = None
    elif mode == "full":
        from torchtitan.distributed.activation_checkpoint import FullAC

        config.activation_checkpoint = FullAC.Config()
    _install()
    return config


def load_std() -> Trainer.Config:
    return _cell("standard")


def load_moonep() -> Trainer.Config:
    return _cell("moonep")
