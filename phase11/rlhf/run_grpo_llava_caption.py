"""Multimodal GRPO RLHF on LLaVA-Pretrain captions, SGLang rollout.

Mirrors `torchtitan/experiments/rl/simple_grpo_sum_digits.py` with two
production-grade swaps:

1. **Generator engine**: ``SGLangGenerator`` instead of
   ``VLLMGenerator``. Same ``Episode`` shape, same ``group_id``
   convention, so GRPO logic is unchanged. SGLang gives us the
   sequence-dim TP shard + RS+AG fusion fabric (proven in
   ``phase11/PHASE11_SGLANG_REPORT.md``) which produces a different
   NCCL trace pattern than vLLM — the comparison is itself a
   research output.

2. **Task**: ``LlavaCaptionTask`` (multimodal) instead of
   ``SumDigitsTask`` (text-only). Each prompt carries an image_path;
   the generator forwards via ``images=`` to SGLang's multimodal
   inference path. Reward = BLEU-1 vs gold caption + length sanity
   + format bonus (verifiable, no separate reward model).

Usage:

    NCCL_DEBUG=INFO NCCL_DEBUG_FILE=phase11/rlhf/trace/nccl-rank-%h-%p.log \\
    NCCL_DEBUG_SUBSYS=COLL \\
    python phase11/rlhf/run_grpo_llava_caption.py \\
        --model-path /root/torchtitan_attention_residual/phase11/hf_aligned_447m_step12500 \\
        --num-steps 50

GPU layout (8 GPUs total):
    ranks 0-3 → PolicyTrainer mesh (FSDP=2 × PP=2 × ...)
    ranks 4-7 → SGLangGenerator mesh (TP=4)
    Grader runs on CPU (rule-based reward, no GPU need).

Cross-mesh Episode + weight transport via Monarch + torchstore.

NOTE: The actual end-to-end run requires a multimodal SGLang model
class (Kimi AttnRes overlay + SigLIP vision tower + projector
registered with SGLang). The current SGLang AttnRes overlay is
text-only; wiring vision is roughly half-day of work — see
``phase11/rlhf/README.md``. This entry-point is structurally
correct and will run once that wiring lands; for now use it with
the text-only path (set ``--text-only`` to skip image input) to
validate the framework + collect the NCCL trace.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make the in-tree task module importable without installing.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import Configurable
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.actors.grader import Grader
from torchtitan.experiments.rl.actors.sglang_generator import SGLangGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.types import Episode
from torchtitan.protocols.model_spec import ModelSpec

from llava_caption_task import LlavaCaptionTask  # noqa: E402

logger = logging.getLogger(__name__)


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes."""

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        return _bootstrap


def _log_samples(episodes: list[Episode], task: LlavaCaptionTask) -> None:
    """Log first sample per group with reward + gold."""
    seen: set[str] = set()
    for ep in episodes:
        if ep.group_id in seen:
            continue
        seen.add(ep.group_id)
        cand = ep.text[:200].replace("\n", " ").strip()
        gold = ep.expected_answer[:120].replace("\n", " ").strip()
        mark = "+" if (ep.reward or 0) > 0 else "-"
        logger.info(f"  [{mark}] reward={ep.reward:+.3f}")
        logger.info(f"       gold: {gold}")
        logger.info(f"       cand: {cand}")


@dataclass(kw_only=True, slots=True)
class _Config(Configurable.Config):
    """Top-level config for the multimodal GRPO trainer."""

    model_spec: Optional[ModelSpec] = None
    hf_assets_path: str = ""
    num_steps: int = 50
    dump_folder: str = "phase11/rlhf/outputs"
    num_episodes_per_step: int = 4
    log_samples: bool = True
    kl_coef: float = 0.05

    llava_json_path: str = "/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
    llava_images_dir: str = "/workspace/.hf_home/LLaVA-Pretrain"
    text_only: bool = False
    """When True, skip image input (lets the framework run on a
    text-only LM ckpt; useful for validating the loop end-to-end
    before the multimodal SGLang model class lands)."""

    trainer: PolicyTrainer.Config = field(default_factory=PolicyTrainer.Config)
    generator: SGLangGenerator.Config = field(default_factory=SGLangGenerator.Config)


async def _async_main(config: _Config) -> None:
    task = LlavaCaptionTask(
        json_path=config.llava_json_path,
        images_dir=config.llava_images_dir,
    )
    logger.info(f"Loaded LlavaCaptionTask with {len(task)} records")

    # Provision GPU meshes — 4 GPUs trainer, 4 GPUs generator.
    provisioner = Provisioner(total_gpus=8)
    trainer_bootstrap = provisioner.allocate(4)
    generator_bootstrap = provisioner.allocate(4)

    # Spawn proc meshes on disjoint GPU partitions (single-node mode).
    trainer_mesh = this_host().spawn_procs(
        per_host={"gpus": 4},
        bootstrap=trainer_bootstrap,
    )
    generator_mesh = this_host().spawn_procs(
        per_host={"gpus": 4},
        bootstrap=generator_bootstrap,
    )
    grader_mesh = this_host().spawn_procs()  # CPU-only mesh

    # torchelastic env must be set per-mesh BEFORE the actors are
    # spawned so the trainer/generator processes inherit
    # RANK/WORLD_SIZE/MASTER_ADDR/etc.
    await setup_torch_elastic_env_async(trainer_mesh)
    await setup_torch_elastic_env_async(generator_mesh)

    trainer = trainer_mesh.spawn(
        "trainer",
        PolicyTrainer,
        config.trainer,
        model_spec=config.model_spec,
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

    # torchstore initialised on the trainer mesh; LocalRankStrategy
    # so colocated procs share a volume (matches upstream simple_grpo).
    await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

    # Warm up: trainer pushes policy v0, generator pulls.
    trainer.push_model_state_dict.call().get()
    generator.pull_model_state_dict.call(0).get()

    for step in range(config.num_steps):
        t0 = time.perf_counter()

        # 1. Sample prompts.
        records = [task.create_question() for _ in range(config.num_episodes_per_step)]
        prompts = [
            f"{task.get_system_prompt()}\n\nUser: {r.prompt_text}\nAssistant:"
            for r in records
        ]
        gold = [r.gold_caption for r in records]
        images = (
            None
            if config.text_only
            else [r.image_path for r in records]
        )

        # 2. Rollout. Pass images positionally per the actor endpoint
        # signature (Monarch ``call`` is not aware of kwargs).
        episodes = generator.generate.call(prompts, gold, images).get()

        # 3. Score.
        episodes = grader.score.call(episodes).get()

        # 4. Compute group-relative advantages (GRPO).
        groups: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            groups[ep.group_id].append(ep)
        for group in groups.values():
            mean_r = sum(ep.reward for ep in group) / len(group)
            std_r = (
                sum((ep.reward - mean_r) ** 2 for ep in group) / max(len(group), 1)
            ) ** 0.5 + 1e-6
            for ep in group:
                ep.advantage = (ep.reward - mean_r) / std_r

        # 5. Trainer step + push new weights.
        # ``trainer.step.call(episodes)`` returns a dict per rank; we
        # take rank-0 metrics.
        metrics = trainer.step.call(episodes).get()
        # Monarch's call().get() returns a ValueMesh; extract rank-0.
        loss = (
            metrics.item(0)["loss"]
            if hasattr(metrics, "item")
            else metrics[0]["loss"]
        )
        trainer.push_model_state_dict.call().get()
        generator.pull_model_state_dict.call(step + 1).get()

        dt = time.perf_counter() - t0
        rewards = [ep.reward for ep in episodes]
        logger.info(
            f"step {step:3d}  loss={loss:.4f}  "
            f"reward_mean={sum(rewards)/len(rewards):+.3f}  "
            f"dt={dt:.1f}s"
        )
        if config.log_samples:
            _log_samples(episodes, task)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Use argparse for our entry-point CLI; the trainer/generator
    # configs themselves are torchtitan ``Configurable``s and load
    # via ConfigManager when invoked through the standard launcher.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True,
                   help="HF-format ckpt dir (e.g. phase11/hf_aligned_447m_step12500)")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--text-only", action="store_true",
                   help="Skip image input (text-only smoke; useful before "
                        "the multimodal SGLang model class lands)")
    p.add_argument("--llava-json",
                   default="/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json")
    p.add_argument("--llava-images",
                   default="/workspace/.hf_home/LLaVA-Pretrain")
    args = p.parse_args()

    config = _Config()
    config.hf_assets_path = args.model_path
    config.num_steps = args.num_steps
    config.text_only = args.text_only
    config.llava_json_path = args.llava_json
    config.llava_images_dir = args.llava_images

    # Trainer + generator parallelism are inherited from default configs.
    # Override here for our 8x 5090 layout: trainer FSDP=4, generator TP=4.
    config.trainer.parallelism.data_parallel_shard_degree = 4
    config.generator.parallelism.tensor_parallel_degree = 4
    config.generator.weight_sync_method = "disk"
    config.generator.weight_sync_disk_path = (
        config.dump_folder + "/sglang_weights"
    )
    Path(config.generator.weight_sync_disk_path).mkdir(parents=True, exist_ok=True)

    asyncio.run(_async_main(config))


if __name__ == "__main__":
    main()
