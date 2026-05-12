"""Multimodal GRPO RLHF on LLaVA-Pretrain captions, 447M Kimi AttnRes
(real research weights, *not* a Qwen3 placeholder), SGLang VLM rollout.

Combines:
  * ``run_grpo_kimi_attn_res.py`` — Kimi 447M Block AttnRes
    model_spec + parallelize adapter + DCP-native load + MonarchRPC
    transport + fp32 MLA fallback (set ATTNRES_MLA_FP32_FALLBACK=1)
  * ``run_grpo_llava_caption.py`` — LlavaCaptionTask (BLEU-1 reward
    against gold caption) + image_data forwarding to the generator

The generator side loads a VLM-format HF ckpt (vision tower +
projector + LM); the trainer side loads the LM-only DCP ckpt and
only updates LM weights. Weight sync via torchstore: vision tower
and projector params don't appear in the LM trainer's state_dict
so they stay frozen on the generator side throughout the run.

Usage:

    PYTHONPATH=$PWD/torchtitan:$PWD ATTNRES_MLA_FP32_FALLBACK=1 \\
    python phase11/rlhf/run_grpo_llava_kimi.py \\
        --dcp-load-path $PWD/phase5/runs/vlm_447m_sft_instruct/checkpoint/step-2344 \\
        --hf-model-path $PWD/phase11/hf/vlm_sft_1ep \\
        --num-steps 500 --num-episodes-per-step 4
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import Configurable
from torchtitan.experiments.rl.actors.grader import Grader
from torchtitan.experiments.rl.actors.sglang_generator import SGLangGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.types import Episode

from llava_caption_task import LlavaCaptionTask  # noqa: E402
from run_grpo_llava_caption import Provisioner  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class _Config(Configurable.Config):
    model_spec: object = None
    hf_assets_path: str = ""
    dcp_initial_load_path: str = ""
    num_steps: int = 500
    dump_folder: str = "phase11/rlhf/outputs/grpo_llava_kimi"
    num_episodes_per_step: int = 4
    log_samples: bool = True
    kl_coef: float = 0.0
    llava_json_path: str = "/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
    llava_images_dir: str = "/workspace/.hf_home/LLaVA-Pretrain"

    trainer: PolicyTrainer.Config = field(default_factory=PolicyTrainer.Config)
    generator: SGLangGenerator.Config = field(default_factory=SGLangGenerator.Config)


def _log_samples(episodes: list[Episode]) -> None:
    seen = set()
    for ep in episodes:
        if ep.group_id in seen:
            continue
        seen.add(ep.group_id)
        cand = ep.text[:200].replace("\n", " ").strip()
        gold = (ep.expected_answer or "")[:120].replace("\n", " ").strip()
        mark = "+" if (ep.reward or 0) > 0 else "-"
        logger.info(f"  [{mark}] reward={ep.reward:+.3f}")
        logger.info(f"       gold: {gold}")
        logger.info(f"       cand: {cand}")


async def _async_main(config: _Config) -> None:
    task = LlavaCaptionTask(
        json_path=config.llava_json_path,
        images_dir=config.llava_images_dir,
    )
    logger.info(f"Loaded LlavaCaptionTask with {len(task)} records")

    provisioner = Provisioner(total_gpus=8)
    trainer_bootstrap = provisioner.allocate(4)
    generator_bootstrap, gen_gpu_ids = provisioner.allocate_shared(4)
    logger.info(f"Generator mesh shares GPUs: {gen_gpu_ids}")

    trainer_mesh = this_host().spawn_procs(
        per_host={"gpus": 4}, bootstrap=trainer_bootstrap,
    )
    generator_mesh = this_host().spawn_procs(
        per_host={"gpus": 4}, bootstrap=generator_bootstrap,
    )
    grader_mesh = this_host().spawn_procs()

    await setup_torch_elastic_env_async(trainer_mesh)
    await setup_torch_elastic_env_async(generator_mesh)

    trainer = trainer_mesh.spawn(
        "trainer",
        PolicyTrainer,
        config.trainer,
        model_spec=config.model_spec,
        hf_assets_path=config.hf_assets_path,
        transfer_dtype=config.generator.model_dtype,
        kl_coef=config.kl_coef,
        dcp_initial_load_path=config.dcp_initial_load_path,
    )
    generator = generator_mesh.spawn(
        "generator",
        SGLangGenerator,
        config.generator,
        model_spec=config.model_spec,
        model_path=config.hf_assets_path,
    )
    grader = grader_mesh.spawn(
        "grader",
        Grader,
        reward_fn=task.reward_function,
    )

    # MonarchRPC transport instead of SHM (container ulimit -l = 64KB).
    # torchstore 0.1.2 removed TransportType / default_transport_type kwarg;
    # MonarchRPC is now the default. Use bare LocalRankStrategy().
    await ts.initialize(
        mesh=trainer_mesh,
        strategy=ts.LocalRankStrategy(),
    )
    trainer.push_model_state_dict.call().get()
    generator.pull_model_state_dict.call(0).get()

    for step in range(config.num_steps):
        t0 = time.perf_counter()

        records = [
            task.create_question() for _ in range(config.num_episodes_per_step)
        ]
        # Embed <image>\n placeholder in the prompt so SGLang's
        # multimodal processor splices vision tokens at that point.
        prompts = [
            f"{task.get_system_prompt()}\n\n<image>\nUser: {r.prompt_text}\nAssistant:"
            for r in records
        ]
        gold = [r.gold_caption for r in records]
        # Read image bytes inline so SGLang doesn't try to mmap from
        # disk via its scheduler-side POSIX SHM bridge (which races
        # against Monarch's actor lifecycle and produces FileNotFoundError
        # on /psm_xxx). Passing raw bytes triggers SGLang's in-RAM
        # path instead of the SHM IPC path.
        import base64
        images = []
        for r in records:
            with open(r.image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            images.append(f"data:image/jpeg;base64,{b64}")

        result = generator.generate.call(prompts, gold, images).get()
        episodes = []
        for _, eps in result:
            episodes.extend(eps)

        result = grader.score.call(episodes).get()
        if isinstance(result, list):
            episodes = result
        else:
            try:
                episodes = next(iter(result))[1]
            except (TypeError, StopIteration):
                episodes = result

        groups: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            groups[ep.group_id].append(ep)
        for group in groups.values():
            mean_r = sum(ep.reward for ep in group) / len(group)
            std_r = (
                sum((ep.reward - mean_r) ** 2 for ep in group)
                / max(len(group), 1)
            ) ** 0.5 + 1e-6
            for ep in group:
                ep.advantage = (ep.reward - mean_r) / std_r

        metrics = trainer.step.call(episodes).get()
        first_metric = None
        try:
            for _, m in metrics:
                first_metric = m
                break
        except TypeError:
            first_metric = metrics
        loss = first_metric.get("loss", 0.0) if isinstance(first_metric, dict) else 0.0
        trainer.push_model_state_dict.call().get()
        generator.pull_model_state_dict.call(step + 1).get()

        dt = time.perf_counter() - t0
        rewards = [ep.reward for ep in episodes]
        logger.info(
            f"step {step:3d}  loss={loss:.4f}  "
            f"reward_mean={sum(rewards)/max(len(rewards),1):+.3f}  "
            f"dt={dt:.1f}s"
        )
        if config.log_samples and step % 5 == 0:
            _log_samples(episodes)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dcp-load-path", required=True,
        help="torchtitan DCP ckpt dir for the LM (no vision tower).",
    )
    p.add_argument(
        "--hf-model-path", required=True,
        help="HF safetensors VLM dir (vision + projector + LM) for SGLang.",
    )
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--num-episodes-per-step", type=int, default=4)
    p.add_argument(
        "--kl-coef", type=float, default=0.0,
        help="0 = vanilla GRPO; >0 engages frozen ref + KL penalty",
    )
    p.add_argument(
        "--flavor", default="kimi_linear_447m_aligned_block_attn_res",
        help="torchtitan flavor name (LM-only Kimi config)",
    )
    args = p.parse_args()

    from torchtitan.experiments.kimi_linear import model_registry as kimi_registry
    from torchtitan.experiments.kimi_linear.parallelize import (
        parallelize_kimi_linear,
    )
    from torchtitan.config import (
        ActivationCheckpointConfig,
        TrainingConfig,
    )
    from torchtitan.protocols.model_converter import ModelConvertersContainer

    model_spec = kimi_registry(args.flavor)
    _orig_parallelize = parallelize_kimi_linear
    _adapter_dump_dir = "phase11/rlhf/outputs/grpo_llava_kimi"

    def _rl_parallelize_adapter(
        model, *, parallel_dims, parallelism, compile_config,
        training=None, model_converters=None,
        ac_config=None, dump_folder=None,
    ):
        if training is None:
            training = TrainingConfig()
        if model_converters is None:
            model_converters = ModelConvertersContainer.Config()
        if ac_config is None:
            ac_config = ActivationCheckpointConfig()
        if dump_folder is None:
            dump_folder = _adapter_dump_dir
        return _orig_parallelize(
            model,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )

    model_spec = model_spec.__class__(
        name=model_spec.name,
        flavor=model_spec.flavor,
        model=model_spec.model,
        parallelize_fn=_rl_parallelize_adapter,
        pipelining_fn=model_spec.pipelining_fn,
        build_loss_fn=model_spec.build_loss_fn,
        post_optimizer_build_fn=model_spec.post_optimizer_build_fn,
        state_dict_adapter=model_spec.state_dict_adapter,
    )
    logger.info(
        f"ModelSpec: name={model_spec.name} flavor={model_spec.flavor}"
    )

    config = _Config()
    config.model_spec = model_spec
    config.hf_assets_path = args.hf_model_path
    config.dcp_initial_load_path = args.dcp_load_path
    config.num_steps = args.num_steps
    config.num_episodes_per_step = args.num_episodes_per_step
    config.kl_coef = args.kl_coef

    config.trainer.parallelism.data_parallel_shard_degree = 4
    config.generator.parallelism.tensor_parallel_degree = 4
    config.generator.gpu_memory_limit = 0.85
    config.generator.weight_sync_method = "disk"
    config.generator.weight_sync_disk_path = (
        config.dump_folder + "/sglang_weights"
    )
    Path(config.generator.weight_sync_disk_path).mkdir(parents=True, exist_ok=True)

    asyncio.run(_async_main(config))


if __name__ == "__main__":
    main()
