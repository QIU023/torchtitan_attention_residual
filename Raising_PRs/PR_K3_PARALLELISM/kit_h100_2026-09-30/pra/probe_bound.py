"""Local probe (never committed; 09-30 H100 copy of kit_overnight_2026-09-29/pra_bound): the Kimi K3 debug model widened to PPMEM_DIM (every width that is a multiple
of dim scales with it, heads by count; 17 layers in blocks of 4, 8 experts, vocab unchanged) on c4 text rows
of up to PPMEM_ROW_TOKENS tokens, FullAC, AdamW(lr=8e-4), type checking off.

Records, per rank:
- every step's peak allocated and reserved memory, taken just before titan resets the peaks;
- in step PPMEM_ACTION_TRACE, after every forward and backward action of the pipeline stages, the block memory
  the pipeline holds (unique storages of the rank store's blocks, the stages' saved block inputs and outputs,
  the tensors of sends issued in this step whose storage is still alive, and live receive buffers), the
  action's own peak, and the AttnRes bound for the blocks alive at that point: "tight" frees a block at the
  backward of the stage that brought it onto the rank, "paper" keeps every block of a micro-batch until its
  last backward on the rank (each block stored once across the rank's virtual stages).
Writes $PPMEM_OUT/rank<r>.json at exit.
"""

import atexit
import json
import os
import weakref

import torch

from torchtitan.trainer import Trainer

_ROW_TOKENS = int(os.environ.get("PPMEM_ROW_TOKENS", "2047"))
_RECORDS: list[dict] = []
_ACTIONS: list[dict] = []
_STAGES: list = []
_SENT: list[tuple[weakref.ref, str]] = []
_TIGHT: dict[tuple[int, int], int] = {}
_PAPER: dict[tuple[int, int], int] = {}
_STORES: list = []

def _process_c4_text_sample(sample, **kwargs):
    from torchtitan.hf_datasets.multimodal.mm_datasets import _process_mm_sample

    out = _process_mm_sample(texts=[sample["text"][: 8 * _ROW_TOKENS]], images=[None], **kwargs)
    if out is None:
        return None
    for key in ("input_ids", "labels", "positions"):
        out[key] = out[key][:_ROW_TOKENS]
    return out


def _not_none(sample):
    return sample is not None


def _storages(tensors) -> dict[int, int]:
    out: dict[int, int] = {}
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda and t.numel():
            st = t.untyped_storage()
            out[st.data_ptr()] = st.nbytes()
    return out


def _flat(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _flat(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _flat(v)


def _account() -> dict:
    store_blocks, deposits, inputs, outputs, recv = {}, {}, {}, {}, {}
    for store in _STORES:
        store_blocks.update(_storages(_flat(getattr(store, "_blocks", {}))))
        deposits.update(_storages(_flat(getattr(store, "_deposits", {}))))
    for stage in _STAGES:
        if stage.is_first:
            continue
        for out_tuple, ins in stage.fwd_cache.values():
            inputs.update(_storages(list(ins)[1:]))
            outputs.update(_storages(list(out_tuple)[1:]))
        for infos in list(stage.args_recv_info.values()) + list(stage.grad_recv_info.values()):
            recv.update(_storages(getattr(i, "buffer", None) for i in (infos or ())))
    hidden_outputs = {}
    for stage in _STAGES:
        if not stage.is_last:
            for out_tuple, _ in stage.fwd_cache.values():
                outputs.update(_storages(list(out_tuple)[1:]))
                hidden_outputs.update(_storages(list(out_tuple)[:1]))
    alive_sent = {"fwd": {}, "bwd": {}, "hidden": {}}
    for ref, kind in _SENT:
        t = ref()
        if t is not None:
            alive_sent[kind].update(_storages([t]))
    blocks = dict(store_blocks)
    blocks.update(inputs)
    blocks.update(outputs)
    blocks.update(alive_sent["fwd"])
    held_elsewhere = set(store_blocks) | set(inputs) | set(outputs) | set(recv)
    sends_only = {k: v for k, v in alive_sent["fwd"].items() if k not in held_elsewhere}
    gib = 2**30
    return {
        "block_gib": sum(blocks.values()) / gib,
        "store_gib": sum(store_blocks.values()) / gib,
        "stage_inputs_gib": sum(v for k, v in inputs.items() if k not in store_blocks) / gib,
        "stage_outputs_gib": sum(v for k, v in outputs.items() if k not in store_blocks and k not in inputs) / gib,
        "fwd_sends_only_gib": sum(sends_only.values()) / gib,
        "bwd_sends_alive_gib": sum(alive_sent["bwd"].values()) / gib,
        "hidden_sends_alive_gib": sum(alive_sent["hidden"].values()) / gib,
        "hidden_sends_only_gib": sum(v for k, v in alive_sent["hidden"].items() if k not in hidden_outputs) / gib,
        "recv_buffers_gib": sum(recv.values()) / gib,
        "deposits_gib": sum(deposits.values()) / gib,
        "tight_bound_gib": sum(_TIGHT.values()) / gib,
        "paper_bound_gib": sum(_PAPER.values()) / gib,
    }


def _brought(stage) -> list[int]:
    layout = stage._layout
    s = stage.stage_index
    return (layout.delta_to_send(s - 1) if s else []) + layout.commits_at(s)


def _first_on_rank(stage) -> bool:
    layout = stage._layout
    mine = [s for s, r in layout.stage_to_rank.items() if r == stage.group_rank]
    return stage.stage_index == min(mine)


def _block_bytes(stage) -> int:
    return int(os.environ.get("PPMEM_SEQ", "2048")) * int(os.environ.get("PPMEM_DIM", "2048")) * 2


def _install() -> None:
    from torch.distributed.pipelining import PipelineStage

    from torchtitan.models.kimi_k3.pipeline_parallel import cache
    from torchtitan.models.kimi_k3.pipeline_parallel.stage import AttnResPipelineStage
    from torchtitan.observability import metrics

    step = int(os.environ.get("PPMEM_ACTION_TRACE", "0"))
    reset = metrics.DeviceMemoryMonitor.reset_peak_stats

    def reset_after_recording(self):
        _RECORDS.append(
            {
                "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
        reset(self)

    metrics.DeviceMemoryMonitor.reset_peak_stats = reset_after_recording

    init = PipelineStage.__init__

    def init_and_register(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _STAGES.append(self)

    PipelineStage.__init__ = init_and_register

    store_init = cache.PPRankLocalCache.__init__

    def store_init_and_register(self, *args, **kwargs):
        store_init(self, *args, **kwargs)
        _STORES.append(self)

    cache.PPRankLocalCache.__init__ = store_init_and_register

    def tracking(kind: str, original):
        # 10-01 fix: a forward send's op 0 carries output 0, the hidden state, which is not a block. It used to be
        # recorded as "fwd", and since _account leaves out_tuple[0] out of the stage outputs, a pending hidden send
        # landed in "sends only" and in block_gib. It is now its own kind, reported as hidden_sends_*.
        def get_ops(self, chunk_id, *args, **kwargs):
            ops = original(self, chunk_id, *args, **kwargs)
            if len(_RECORDS) == step:
                for i, op in enumerate(ops):
                    t = getattr(op, "tensor", None)
                    if not isinstance(t, torch.Tensor):
                        continue
                    if kind == "fwd" and i == 0:
                        assert t.data_ptr() == self.fwd_cache[chunk_id][0][0].data_ptr(), "op 0 is not output 0"
                        _SENT.append((weakref.ref(t), "hidden"))
                    else:
                        _SENT.append((weakref.ref(t), kind))
            return ops

        return get_ops

    PipelineStage.get_fwd_send_ops = tracking("fwd", PipelineStage.get_fwd_send_ops)
    PipelineStage.get_bwd_send_ops = tracking("bwd", PipelineStage.get_bwd_send_ops)

    def traced(kind: str, original):
        def action(self, chunk_id, *args, **kwargs):
            on = len(_RECORDS) == step and self.has_backward
            if on:
                torch.cuda.reset_peak_memory_stats()
            out = original(self, chunk_id, *args, **kwargs)
            if not self.has_backward:
                return out
            key_bytes = _block_bytes(self)
            if kind == "F":
                for b in _brought(self):
                    _TIGHT[(chunk_id, b)] = key_bytes
                    _PAPER[(chunk_id, b)] = key_bytes
            else:
                for b in _brought(self):
                    _TIGHT.pop((chunk_id, b), None)
                if _first_on_rank(self):
                    for key in [k for k in _PAPER if k[0] == chunk_id]:
                        _PAPER.pop(key)
            if on:
                record = {
                    "action": f"{self.stage_index}{kind}{chunk_id}",
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "after_gib": torch.cuda.memory_allocated() / 2**30,
                }
                record.update(_account())
                _ACTIONS.append(record)
            return out

        return action

    AttnResPipelineStage.forward_one_chunk = traced("F", AttnResPipelineStage.forward_one_chunk)
    AttnResPipelineStage.backward_one_chunk = traced("B", AttnResPipelineStage.backward_one_chunk)

    def dump() -> None:
        out = os.environ.get("PPMEM_OUT")
        if not out or not torch.cuda.is_available():
            return
        _RECORDS.append(
            {
                "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
        rank = int(os.environ.get("RANK", "0"))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
            json.dump({"rank": rank, "records": _RECORDS, "actions": _ACTIONS}, f)

    atexit.register(dump)
    _install_grad_dump()


def _install_grad_dump() -> None:
    """10-01: with PPMEM_GRAD_DUMP=<dir>, write every parameter's step-1 gradient hash per rank, before clipping."""
    out = os.environ.get("PPMEM_GRAD_DUMP")
    if not out:
        return
    import hashlib

    from torchtitan.training_engine import TrainingEngine

    step = TrainingEngine.optim_step

    def dump_then_step(self):
        if self.num_completed_steps == 0:
            record = {}
            for part in self.model_parts:
                for name, param in part.named_parameters():
                    grad = param.grad
                    if grad is None:
                        record[name] = None
                        continue
                    if hasattr(grad, "to_local"):
                        grad = grad.to_local()
                    flat = grad.detach().contiguous().reshape(-1)
                    record[name] = {
                        "dtype": str(flat.dtype),
                        "shape": list(grad.shape),
                        "sha256": hashlib.sha256(flat.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                        "norm": float(flat.float().norm()),
                    }
            rank = int(os.environ.get("RANK", "0"))
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
                json.dump(record, f)
            print(f"PPMEM_GRAD_DUMP rank {rank}: {len(record)} parameters", flush=True)
        return step(self)

    TrainingEngine.optim_step = dump_then_step


def _widened_model(dim: int, seq: int):
    import inspect

    from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config

    s = dim // 256
    assert dim == 256 * s
    sp = {"enable_sp": False} if "enable_sp" in inspect.signature(_kimi_k3_config).parameters else {}
    return _kimi_k3_config(
        **sp,
        max_context_length=seq,
        dim=dim,
        vocab_size=2048,
        num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16},
        attn_res_block_size=4,
        num_heads=4 * s,
        q_lora_rank=128 * s,
        kv_lora_rank=64 * s,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=512 * s,
        latent_dim=128 * s,
        expert_hidden_dim=128 * s,
        num_experts=8,
        top_k=2,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=256 * s,
            qkv_dim=512 * s,
            hidden_dim=512 * s,
            num_layers=2,
            num_heads=4 * s,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend="flex",
    )


def lb_probe() -> Trainer.Config:
    from torchtitan.components.data import SingleDatasetConfig
    from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.hf_datasets.multimodal.mm_datasets import MultiModalProcessor
    from torchtitan.models.kimi_k3.config_registry import (
        _kimi_k3_multimodal_dataloader,
        kimi_k3_debugmodel,
    )
    from torchtitan_recipes.tests import _set_spmd_typechecking

    dim = int(os.environ.get("PPMEM_DIM", "2048"))
    seq = int(os.environ.get("PPMEM_SEQ", "2048"))
    config = kimi_k3_debugmodel(seq_len=seq)
    _set_spmd_typechecking(config, typechecking=False)
    from torchtitan.components.optim import AdamW, OptimizersContainer

    config.optim.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    config.activation_checkpoint = FullAC.Config()
    config.model = _widened_model(dim, seq)
    config.dataloader = _kimi_k3_multimodal_dataloader(
        SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                split="train",
                load_dataset_kwargs={"data_files": "tests/assets/c4_test/data.json"},
            ),
            processor=MultiModalProcessor.Config(sample_processor=_process_c4_text_sample),
            post_filters=(_not_none,),
        )
    )
    stages = os.environ.get("PPMEM_STAGES", "")
    if stages:
        from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part
        from torchtitan.models.kimi_k3.model import KimiK3Model

        p = config.parallelism
        split = _generate_llm_fqn_per_model_part(
            int(stages), len(config.model.layers), p.pipeline_parallel_first_stage_less_layers,
            p.pipeline_parallel_last_stage_less_layers)
        split[0][:0] = KimiK3Model.pipeline_first_stage_module_fqns
        split[-1].extend(KimiK3Model.pipeline_last_stage_module_fqns)
        p.pipeline_parallel_module_fqns_per_model_part = split
    _install()
    return config
