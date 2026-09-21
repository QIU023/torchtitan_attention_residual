"""LOCAL PROBE HACKS for the round-3 reference matrix (never committed). argv[1] = tree.

Adds to kimi_k3/config_registry.py the c4 text-row flavor and its naive / explicit-split
variants, MB_REVERSE to the trainer, NOSYNC_GA to the engine, GN_FP32 to clip_grad_norm_,
and the SM120 guard lift. Undo with: git checkout -- <the four files> in the tree.
"""

import pathlib
import subprocess
import sys

tree = pathlib.Path(sys.argv[1])

# 1. flavors
p = tree / "torchtitan/models/kimi_k3/config_registry.py"
s = p.read_text()
if "PROBE ONLY" not in s:
    s += '''

# ---------------------------------------------------------------------------
# PROBE ONLY (not committed): the c4 text-row flavor and its pipeline variants.
# ---------------------------------------------------------------------------
_C4_ROW_TOKENS = 256  # one row per 256-token micro-batch


def _process_c4_text_sample(sample, **kwargs):  # PROBE ONLY (not committed)
    """A c4 doc as one text-only row: its first 256 tokens (the multimodal
    batcher packs whole rows into a micro-batch, so a row never spans two)."""
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
    import functools

    from torchtitan.models.kimi_k3.pipeline_parallel import pipeline_kimi_k3

    assert config.model_spec is not None
    config.model_spec.pipelining_fn = functools.partial(pipeline_kimi_k3, attn_res_cache=False)
    return config


def _eight_stages(config: Trainer.Config) -> Trainer.Config:  # PROBE ONLY (not committed)
    # pp2 x vp4: no layers_per_stage reaches 8 stages for 19 units, so the split is spelled out
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = [
        ["vision_encoder", "tok_embeddings", "layers.0", "layers.1"],
        ["layers.2", "layers.3", "layers.4"],
        ["layers.5", "layers.6", "layers.7"],
        ["layers.8", "layers.9"],
        ["layers.10", "layers.11"],
        ["layers.12", "layers.13"],
        ["layers.14", "layers.15"],
        ["layers.16", "norm", "lm_head", "output_res_proj", "output_res_norm"],
    ]
    return config


def kimi_k3_debugmodel_c4_pp_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_8stages() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _eight_stages(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_8stages_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(_eight_stages(kimi_k3_debugmodel_c4()))


def kimi_k3_debugmodel_c4_seed() -> Trainer.Config:  # PROBE ONLY (not committed)
    from torchtitan.components.checkpointer import CheckpointManager

    config = kimi_k3_debugmodel_c4()
    config.checkpointer = CheckpointManager.Config()
    config.create_seed_checkpoint = True
    return config
'''
    p.write_text(s)

# 2. MB_REVERSE in the trainer
p = tree / "torchtitan/trainer.py"
s = p.read_text()
anchor = "            microbatch_groups.append(microbatch_group)\n        sl.log_trace_scalar({\"local_valid_tokens\": local_valid_tokens})\n"
assert s.count(anchor) == 1 or "MB_REVERSE" in s
s = s.replace(
    anchor,
    "            microbatch_groups.append(microbatch_group)\n"
    "        if os.environ.get(\"MB_REVERSE\") == \"1\":  # LOCAL PROBE HACK (not committed): accumulation order only\n"
    "            microbatch_groups.reverse()\n"
    "        sl.log_trace_scalar({\"local_valid_tokens\": local_valid_tokens})\n",
)
p.write_text(s)

# 3. NOSYNC_GA in the engine
p = tree / "torchtitan/training_engine.py"
s = p.read_text()
anchor = "                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]\n"
assert s.count(anchor) == 1
s = s if "NOSYNC_GA" in s else s.replace(
    anchor,
    anchor
    + "\n        if __import__(\"os\").environ.get(\"NOSYNC_GA\") == \"1\" and not self.parallel_dims.pp_enabled:  # LOCAL PROBE HACK (not committed)\n"
    "            # Accumulate micro-batches the way torch pipelining does under FSDP: no sync until the last backward.\n"
    "            _last = accumulation_index == self.num_accumulation_steps - 1\n"
    "            for _part in self.model_parts:\n"
    "                _part.set_is_last_backward(_last)\n"
    "                _part.set_reshard_after_backward(_last)\n"
    "                _part.set_requires_gradient_sync(_last)\n",
)
p.write_text(s)

# 4. GN_FP32
p = tree / "torchtitan/distributed/utils.py"
s = p.read_text()
anchor = "    grads = [p.grad for p in parameters if p.grad is not None]\n    total_norm = torch.nn.utils.get_total_norm(\n"
assert s.count(anchor) == 1 or "GN_FP32" in s
s = s if "GN_FP32" in s else s.replace(
    anchor,
    "    grads = [p.grad for p in parameters if p.grad is not None]\n"
    "    if __import__(\"os\").environ.get(\"GN_FP32\") == \"1\":  # LOCAL PROBE HACK (not committed): norm in float32\n"
    "        grads = [g.float() for g in grads]\n"
    "    total_norm = torch.nn.utils.get_total_norm(\n",
)
p.write_text(s)

# 5. the SM120 guard lift
patch = pathlib.Path(__file__).with_name("kda_guard_lift.patch")
if subprocess.run(["git", "-C", str(tree), "apply", "--check", str(patch)], capture_output=True).returncode == 0:
    subprocess.run(["git", "-C", str(tree), "apply", str(patch)], check=True)
print("probe hacks applied to", tree)
