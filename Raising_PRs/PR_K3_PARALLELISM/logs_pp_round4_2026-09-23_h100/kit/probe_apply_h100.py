"""LOCAL PROBE HACKS for the PR 4312 body tables on the H100, for the post-#4810 head (never committed).
argv[1] = tree. Undo with: git -C <tree> checkout -- .

The c4 text-row flavor and its seed flavor; the naive transport through an environment switch, because
after #4810 the model owns pipeline() and there is no pipelining_fn for a partial; the 8- and 16-stage
splits spelled out (the PR's own pp2 x vp4 / pp4 x vp4 shapes); MB_REVERSE, NOSYNC_GA, GN_FP32, FP32_PROBE,
the reference logging its step loss the way the pipeline's last stage does, and the KDA capability guard lifted for SM90.
"""
import pathlib
import sys

tree = pathlib.Path(sys.argv[1])


def patch(rel, old, new, marker):
    p = tree / rel
    s = p.read_text()
    if marker in s:
        return
    assert s.count(old) == 1, (rel, s.count(old))
    p.write_text(s.replace(old, new, 1))


# 1. flavors
p = tree / "torchtitan/models/kimi_k3/config_registry.py"
s = p.read_text()
if "PROBE ONLY" not in s:
    s += '''

# ---------------------------------------------------------------------------
# PROBE ONLY (not committed): the c4 text-row flavor and its pipeline variants.
# ---------------------------------------------------------------------------
_C4_ROW_TOKENS = 256  # one row per 256-token micro-batch

_EIGHT_STAGES = [  # the PR's pp2 x vp4 split (tests/unit_tests/cpu/test_kimi_k3_pp_layout.py)
    ["vision_encoder", "tok_embeddings", "layers.0", "layers.1"],
    ["layers.2", "layers.3", "layers.4"],
    ["layers.5", "layers.6", "layers.7"],
    ["layers.8", "layers.9"],
    ["layers.10", "layers.11"],
    ["layers.12", "layers.13"],
    ["layers.14", "layers.15"],
    ["layers.16", "norm", "lm_head", "output_res_proj", "output_res_norm"],
]


def _process_c4_text_sample(sample, **kwargs):  # PROBE ONLY (not committed)
    """A c4 doc as one text-only row: its first 256 tokens."""
    from torchtitan.hf_datasets.multimodal.mm_datasets import _process_mm_sample

    out = _process_mm_sample(texts=[sample["text"][:4000]], images=[None], **kwargs)
    if out is None:
        return None
    n = _C4_ROW_TOKENS - 1
    for key in ("input_ids", "labels", "positions"):
        out[key] = out[key][:n]
    return out


def kimi_k3_debugmodel_c4() -> Trainer.Config:  # PROBE ONLY (not committed)
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.data.sources import HuggingFaceRandomAccessSource

    config = kimi_k3_debugmodel()
    # Every cell loads the seed checkpoint from <dump_folder>/checkpoint and never saves.
    config.checkpointer = CheckpointManager.Config(interval=100000)
    config.dataloader = _kimi_k3_multimodal_dataloader(
        SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                split="train",
                load_dataset_kwargs={"data_files": "tests/assets/c4_test/data.json"},
            ),
            processor=MultiModalProcessor.Config(sample_processor=_process_c4_text_sample),
            post_filters=(lambda sample: sample is not None,),
        )
    )
    return config


def _naive(config: Trainer.Config) -> Trainer.Config:  # PROBE ONLY (not committed)
    import os

    os.environ["ATTN_RES_NAIVE"] = "1"  # read by the probe hook in KimiK3Model.pipeline
    return config


def _eight_stages(config: Trainer.Config) -> Trainer.Config:  # PROBE ONLY (not committed)
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = [list(s) for s in _EIGHT_STAGES]
    return config


def _sixteen_stages(config: Trainer.Config) -> Trainer.Config:  # PROBE ONLY (not committed)
    from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp4_vp4

    config.parallelism.pipeline_parallel_module_fqns_per_model_part = (
        kimi_k3_debugmodel_pp4_vp4().parallelism.pipeline_parallel_module_fqns_per_model_part
    )
    return config


def kimi_k3_debugmodel_c4_pp_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_8stages() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _eight_stages(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_8stages_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(_eight_stages(kimi_k3_debugmodel_c4()))


def kimi_k3_debugmodel_c4_16stages() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _sixteen_stages(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_16stages_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(_sixteen_stages(kimi_k3_debugmodel_c4()))


def kimi_k3_debugmodel_c4_seed() -> Trainer.Config:  # PROBE ONLY (not committed)
    from torchtitan.components.checkpointer import CheckpointManager

    config = kimi_k3_debugmodel_c4()
    config.checkpointer = CheckpointManager.Config()
    config.create_seed_checkpoint = True
    return config
'''
    p.write_text(s)

# 2. the naive transport switch, where the model owns its pipelining
patch(
    "torchtitan/models/kimi_k3/model.py",
    "        return pipeline_kimi_k3(self, **kwargs)\n",
    '        return pipeline_kimi_k3(self, attn_res_cache=__import__("os").environ.get("ATTN_RES_NAIVE") != "1", **kwargs)  # LOCAL PROBE HACK (not committed)\n',
    "ATTN_RES_NAIVE",
)

# 3. MB_REVERSE
patch(
    "torchtitan/trainer.py",
    "            microbatch_groups.append(microbatch_group)\n        sl.log_trace_scalar({\"local_valid_tokens\": local_valid_tokens})\n",
    "            microbatch_groups.append(microbatch_group)\n"
    "        if os.environ.get(\"MB_REVERSE\") == \"1\":  # LOCAL PROBE HACK (not committed): accumulation order only\n"
    "            microbatch_groups.reverse()\n"
    "        sl.log_trace_scalar({\"local_valid_tokens\": local_valid_tokens})\n",
    "MB_REVERSE",
)

# 4. NOSYNC_GA
patch(
    "torchtitan/training_engine.py",
    "                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]\n",
    "                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]\n"
    "\n        if __import__(\"os\").environ.get(\"NOSYNC_GA\") == \"1\" and not self.parallel_dims.pp_enabled:  # LOCAL PROBE HACK (not committed)\n"
    "            # Accumulate micro-batches the way torch pipelining does under FSDP: no sync until the last backward.\n"
    "            _last = accumulation_index == self.num_accumulation_steps - 1\n"
    "            for _part in self.model_parts:\n"
    "                _part.set_is_last_backward(_last)\n"
    "                _part.set_reshard_after_backward(_last)\n"
    "                _part.set_requires_gradient_sync(_last)\n",
    "NOSYNC_GA",
)

# 5. GN_FP32
patch(
    "torchtitan/distributed/utils.py",
    "    grads = [p.grad for p in parameters if p.grad is not None]\n    total_norm = torch.nn.utils.get_total_norm(\n",
    "    grads = [p.grad for p in parameters if p.grad is not None]\n"
    "    if __import__(\"os\").environ.get(\"GN_FP32\") == \"1\":  # LOCAL PROBE HACK (not committed): norm in float32\n"
    "        grads = [g.float() for g in grads]\n"
    "    total_norm = torch.nn.utils.get_total_norm(\n",
    "GN_FP32",
)

# 6. FP32_PROBE: bf16 casts stay float32, experts looped, KDA on the reference implementation
patch(
    "torchtitan/trainer.py",
    "import torch\nimport tyro\n",
    '''import torch

if __import__("os").environ.get("FP32_PROBE") == "1":  # LOCAL PROBE HACK (not committed): bf16 casts stay float32
    torch.Tensor.bfloat16 = lambda self, *a, **k: self
    _fp32_probe_to = torch.Tensor.to

    def _fp32_probe_to_fn(self, *a, **k):
        a = tuple(torch.float32 if x is torch.bfloat16 else x for x in a)
        if k.get("dtype") is torch.bfloat16:
            k["dtype"] = torch.float32
        return _fp32_probe_to(self, *a, **k)

    torch.Tensor.to = _fp32_probe_to_fn
import tyro
''',
    "FP32_PROBE",
)
patch(
    "torchtitan/models/common/moe.py",
    "        return torch._grouped_mm(A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs)\n",
    '''        if __import__("os").environ.get("FP32_PROBE") == "1":  # LOCAL PROBE HACK (not committed): per-expert loop
            W, pieces, start = weight_EOI.transpose(-2, -1), [], 0
            for e, end in enumerate(offs.tolist()):
                pieces.append(A[start:end] @ W[e].to(A.dtype))
                start = end
            pieces.append(A.new_zeros(A.shape[0] - start, W.shape[-1]))
            return torch.cat(pieces)
        return torch._grouped_mm(A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs)
''',
    "FP32_PROBE",
)
patch(
    "torchtitan/models/kimi_k3/kda.py",
    '            impl="fused",\n',
    '            impl="reference" if __import__("os").environ.get("FP32_PROBE") == "1" else "fused",  # LOCAL PROBE HACK (not committed)\n',
    "FP32_PROBE",
)
patch(
    "torchtitan/models/kimi_k3/kda.py",
    "            cu_seqlens=cu_seqlens,\n        )\n        return output_1THV\n",
    '            cu_seqlens=cu_seqlens,\n            **({"impl": "reference"} if __import__("os").environ.get("FP32_PROBE") == "1" else {}),  # LOCAL PROBE HACK (not committed)\n        )\n        return output_1THV\n',
    '{"impl": "reference"}',
)

# 7. the reference logs its step loss the way the pipeline's last stage does
patch(
    "torchtitan/trainer.py",
    """            if should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)
""",
    """            if should_log and os.environ.get("NOSYNC_GA") == "1":  # LOCAL PROBE HACK (not committed): the reference logs its step loss the way the pipeline's last stage does, torch.sum(torch.stack(losses)), not a running add
                _probe_losses = globals().setdefault("_PROBE_LOSSES", [])
                _probe_losses.append(detached_loss.clone())
                if fwd_bwd_index == len(microbatch_groups) - 1:
                    accumulated_loss = torch.sum(torch.stack(_probe_losses)).to(detached_loss.device)
                    _probe_losses.clear()
            elif should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)
""",
    "_PROBE_LOSSES",
)
# 8. the KDA capability guard lifted: attn-gym routes SM90 through portable kernels, as on the 09-21 H200 run
patch(
    "torchtitan/models/kimi_k3/kda.py",
    """        capability = torch.cuda.get_device_capability(q_1THK.device)
        if capability not in {(10, 0), (10, 3)}:
            raise RuntimeError(
                "Attention Gym KDA requires Blackwell SM100/SM103; "
                f"got CUDA capability {capability}."
            )

""",
    "        # LOCAL PROBE HACK (not committed): the capability guard lifted for SM90\n",
    "guard lifted for SM90",
)
print("H100 probe hacks applied to", tree)
