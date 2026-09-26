"""Scratch flavor for the PP lower-bound campaign (never committed), from kit_pp_optimize_2026-09-24.

Adds to that probe: the rank store's live bytes (unique storages it references), with a
high-water mark per step, and PPMEM_LPS for pipeline_parallel_layers_per_stage. Set
PPMEM_STORE_TRACK=0 for timing runs, since the accounting walks the store on every call.

Original notes:

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


_STORE_PEAK = [0]
_STORES: list = []


def _store_bytes(store) -> int:
    seen, total = set(), 0

    def visit(obj) -> None:
        nonlocal total
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                storage = obj.untyped_storage()
                if storage.data_ptr() not in seen:
                    seen.add(storage.data_ptr())
                    total += storage.nbytes()
        elif isinstance(obj, dict):
            for v in obj.values():
                visit(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                visit(v)

    for name in ("_rows", "_blocks", "_deposits"):
        visit(getattr(store, name, None))
    return total


def _track_store() -> None:
    from torchtitan.models.kimi_k3.pipeline_parallel import cache

    cls = cache.PPRankLocalCache
    for name in ("allocate", "put", "mark", "release", "deposit", "collect"):
        original = getattr(cls, name, None)
        if original is None:
            continue

        def tracked(self, *args, _original=original, **kwargs):
            out = _original(self, *args, **kwargs)
            if not any(store is self for store in _STORES):
                _STORES.append(self)
            _STORE_PEAK[0] = max(_STORE_PEAK[0], _store_bytes(self))
            return out

        setattr(cls, name, tracked)


def _manager_pinned_peak() -> int:
    peak = 0
    for store in _STORES:
        storage = getattr(store, "_storage", None)
        if storage is not None:
            peak = max(peak, int(storage.stats["pinned_peak_bytes"]))
            storage.stats["pinned_peak_bytes"] = storage.stats["pinned_bytes"]
    return peak


_ACTIONS: list = []
_PLAN_INPUTS: list = []


def _capture_plan_inputs() -> None:
    """Keep what MemoryPlan was built from, to replay the plan offline."""
    try:
        from torchtitan.models.kimi_k3.pipeline_parallel import activations
    except ImportError:
        return
    plan_cls = getattr(activations, "MemoryPlan", None)
    if plan_cls is None:
        return
    init = plan_cls.__init__

    def captured(self, orders, profiles, **kwargs):
        _PLAN_INPUTS.append((orders, profiles, kwargs))
        init(self, orders, profiles, **kwargs)

    plan_cls.__init__ = captured


def _trace_actions() -> None:
    """PPMEM_ACTION_TRACE=<step>: in that step, each compute action's peak, the memory after it and
    the rank store's bytes, written to $PPMEM_OUT/actions<r>.json at exit. Resets the peak per
    action, so that step's per-step record is not a step peak."""
    from torchtitan.models.kimi_k3.pipeline_parallel.stage import AttnResPipelineStage

    step = int(os.environ["PPMEM_ACTION_TRACE"])
    seen = [0]

    def wrap(name: str, kind: str) -> None:
        original = getattr(AttnResPipelineStage, name)

        def traced(self, chunk_id, *args, _original=original, **kwargs):
            # _RECORDS[0] is titan's reset before step 1, so step k runs with k records
            on = self.has_backward and len(_RECORDS) == step
            if on:
                torch.cuda.reset_peak_memory_stats()
            out = _original(self, chunk_id, *args, **kwargs)
            if on:
                store = self._store
                _ACTIONS.append(
                    {
                        "action": f"{self.stage_index}{kind}{chunk_id}",
                        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "after_gib": torch.cuda.memory_allocated() / 2**30,
                        "store_gib": (_store_bytes(store) if store is not None else 0) / 2**30,
                    }
                )
            return out

        setattr(AttnResPipelineStage, name, traced)

    wrap("forward_one_chunk", "F")
    wrap("backward_one_chunk", "B")


def _record_peak() -> None:
    stats = torch.cuda.memory_stats()
    _RECORDS.append(
        {
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "allocated_now_gib": torch.cuda.memory_allocated() / 2**30,
            "recv_buffers_gib": _recv_buffer_bytes() / 2**30,
            "store_peak_gib": _STORE_PEAK[0] / 2**30,
            "pinned_peak_gib": _manager_pinned_peak() / 2**30,
            "alloc_retries": stats.get("num_alloc_retries", 0),
            "num_ooms": stats.get("num_ooms", 0),
        }
    )
    _STORE_PEAK[0] = 0


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
    if os.environ.get("PPMEM_STORE_TRACK", "1") == "1":
        _track_store()
    if os.environ.get("PPMEM_ACTION_TRACE"):
        _trace_actions()
    _capture_plan_inputs()


def _dump_peak_memory() -> None:
    out = os.environ.get("PPMEM_OUT")
    if not out or not torch.cuda.is_available():
        return
    _record_peak()
    rank = int(os.environ.get("RANK", "0"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
        json.dump({"rank": rank, "records": _RECORDS}, f)
    if _ACTIONS:
        with open(os.path.join(out, f"actions{rank}.json"), "w") as f:
            json.dump({"rank": rank, "actions": _ACTIONS}, f)
    if _PLAN_INPUTS:
        import pickle

        orders, profiles, kwargs = _PLAN_INPUTS[0]
        with open(os.path.join(out, f"plan_inputs{rank}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "orders": {
                        r: [None if a is None else str(a) for a in acts]
                        for r, acts in orders.items()
                    },
                    "profiles": profiles,
                    "kwargs": kwargs,
                },
                f,
            )
    memory = next(
        (s._memory for s in _STAGES if getattr(s, "_memory", None) is not None), None
    )
    plan = getattr(memory, "plan", None)
    if plan is not None:
        storage = memory._storage
        with open(os.path.join(out, f"plan{rank}.json"), "w") as f:
            json.dump(
                {
                    "summary": {r: plan.summary(r) for r in sorted(plan.profiled)},
                    "dests": plan.dests,
                    "spans_gib": {r: n / 2**30 for r, n in plan.spans.items()},
                    "moved": sorted(f"{r}:{s}:{m}:{b}" for (r, s, m), b in plan.backend.items()),
                    "profiled_peak_gib": plan.profiled[rank] / 2**30,
                    "planned_peak_gib": plan.peaks[rank] / 2**30,
                    "stats": dict(storage.stats) if storage is not None else {},
                },
                f,
                indent=1,
            )


def lb_probe() -> Trainer.Config:
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
    if os.environ.get("PPMEM_LPS"):
        config.parallelism.pipeline_parallel_layers_per_stage = int(
            os.environ["PPMEM_LPS"]
        )
    if os.environ.get("PPMEM_AC", "full") == "full":
        config.activation_checkpoint = FullAC.Config()
    else:
        config.activation_checkpoint = None
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
    memory = getattr(config.model, "pp_memory", None)
    switches = ("PPMEM_MANAGER", "PPMEM_OFFLOAD", "PPMEM_BALANCE", "PPMEM_TARGET_GIB")
    if memory is None and any(os.environ.get(name) for name in switches):
        raise ValueError("this tree's model config has no pp_memory")
    if memory is not None and os.environ.get("PPMEM_MANAGER") == "1":
        memory.manager = True
    if memory is not None and os.environ.get("PPMEM_OFFLOAD") == "1":
        memory.offload = True
    if memory is not None and os.environ.get("PPMEM_BALANCE") == "1":
        memory.balance = True
    if memory is not None and os.environ.get("PPMEM_TARGET_GIB"):
        memory.target_gib = float(os.environ["PPMEM_TARGET_GIB"])
    if memory is not None and os.environ.get("PPMEM_HOST_GBPS"):
        memory.host_gbps = float(os.environ["PPMEM_HOST_GBPS"])
    if memory is not None and os.environ.get("PPMEM_PEER_GBPS"):
        memory.peer_gbps = float(os.environ["PPMEM_PEER_GBPS"])
    parked = int(os.environ.get("PPMEM_OFFLOAD_MB", "0"))
    if parked:
        from torchtitan.models.kimi_k3.pipeline_parallel.activations import PPOffloadKnobs

        config.model.pp_offload = PPOffloadKnobs(
            microbatches=parked, lead=int(os.environ.get("PPMEM_OFFLOAD_LEAD", "1"))
        )
    pairs = os.environ.get("PPMEM_BALANCE_PAIRS", "")
    if pairs:
        from torchtitan.models.kimi_k3.pipeline_parallel.activations import PPBalanceKnobs

        config.model.pp_balance = PPBalanceKnobs(
            pairs=tuple(tuple(int(x) for x in p.split(":")) for p in pairs.split(",")),
            microbatches=int(os.environ.get("PPMEM_BALANCE_MB", "8")),
            lead=int(os.environ.get("PPMEM_BALANCE_LEAD", "1")),
            pool_gib=float(os.environ.get("PPMEM_BALANCE_POOL_GIB", "2")),
        )
    _install_hooks()
    atexit.register(_dump_peak_memory)
    return config
