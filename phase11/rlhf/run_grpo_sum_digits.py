"""Production-grade GRPO RL on sum-digits, via SGLang generator.

Mirrors ``torchtitan/experiments/rl/simple_grpo_sum_digits.py`` but
swaps the rollout engine from vLLM to SGLang via the upstream-RFC
``SGLangGenerator``. Same task (deterministic, verifiable reward),
same trainer (PolicyTrainer + Qwen3-0.6B), same trace pipeline.

Usage:

    bash phase11/rlhf/run_grpo_sum_digits_with_trace.sh

GPU layout (8× RTX 5090):
    ranks 0-3 → PolicyTrainer (FSDP=4)
    ranks 4-7 → SGLangGenerator (TP=4, lead/follower; rank 4 is the
                lead that holds the Engine, ranks 5-7 idle)
    Grader   → CPU mesh (rule-based reward)

NCCL trace from this run is the closest analog to upstream's
vLLM-based RL loop, useful for fabric-pattern comparison
(``phase11/rlhf/trace_grpo_qwen3_*``).
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import Configurable
from torchtitan.config.configs import DebugConfig, ParallelismConfig
from torchtitan.experiments.rl.actors.grader import Grader
from torchtitan.experiments.rl.actors.sglang_generator import SGLangGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.sum_digits import extract_answer, SumDigitsTask
from torchtitan.experiments.rl.types import Episode

# Reuse the Provisioner from the LLaVA entry-point.
from run_grpo_llava_caption import Provisioner  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class _Config(Configurable.Config):
    model_spec: object = None
    hf_assets_path: str = ""
    num_steps: int = 50
    dump_folder: str = "phase11/rlhf/outputs/grpo_sum_digits"
    num_episodes_per_step: int = 4
    log_samples: bool = True
    kl_coef: float = 0.0

    trainer: PolicyTrainer.Config = field(default_factory=PolicyTrainer.Config)
    generator: SGLangGenerator.Config = field(default_factory=SGLangGenerator.Config)


def _log_samples(episodes: list[Episode]) -> None:
    seen = set()
    for ep in episodes:
        if ep.group_id in seen:
            continue
        seen.add(ep.group_id)
        extracted = extract_answer(ep.text)
        is_correct = (
            extracted == int(ep.expected_answer) if ep.expected_answer else None
        )
        mark = "+" if is_correct else "-"
        logger.info(
            f"  [{mark}] expected={ep.expected_answer} extracted={extracted} "
            f"reward={ep.reward:+.2f}"
        )
        logger.info(f"       {ep.text[:240].replace(chr(10), ' ').strip()}")


async def _async_main(config: _Config) -> None:
    task = SumDigitsTask(seed=42)

    provisioner = Provisioner(total_gpus=8)
    trainer_bootstrap = provisioner.allocate(4)
    generator_bootstrap, gen_gpu_ids = provisioner.allocate_shared(4)
    logger.info(f"Generator mesh shares GPUs: {gen_gpu_ids}")

    trainer_mesh = this_host().spawn_procs(per_host={"gpus": 4}, bootstrap=trainer_bootstrap)
    generator_mesh = this_host().spawn_procs(per_host={"gpus": 4}, bootstrap=generator_bootstrap)
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

    await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())
    trainer.push_model_state_dict.call().get()
    generator.pull_model_state_dict.call(0).get()

    for step in range(config.num_steps):
        t0 = time.perf_counter()

        # 1. Sample prompts.
        records = []
        for _ in range(config.num_episodes_per_step):
            q, ans = task.create_question()
            records.append((q, ans))
        prompts = [
            f"{task.get_system_prompt()}\n\nUser: {q}\nAssistant:"
            for q, _ in records
        ]
        gold = [a for _, a in records]

        # 2. Rollout.
        result = generator.generate.call(prompts, gold, None).get()
        episodes = []
        for _, eps in result:
            episodes.extend(eps)

        # 3. Score.
        result = grader.score.call(episodes).get()
        if isinstance(result, list):
            episodes = result
        else:
            try:
                episodes = next(iter(result))[1]
            except (TypeError, StopIteration):
                episodes = result

        # 4. GRPO advantage.
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

        # 5. Trainer step.
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
    p.add_argument("--model-path", required=True)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--num-episodes-per-step", type=int, default=4)
    args = p.parse_args()

    from torchtitan.models.qwen3 import model_registry as qwen3_registry
    from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3

    flavor = os.environ.get("RLHF_FLAVOR", "0.6B_varlen")
    model_spec = qwen3_registry(flavor)
    model_spec.parallelize_fn = parallelize_qwen3
    logger.info(f"Loaded ModelSpec: name={model_spec.name} flavor={model_spec.flavor}")

    config = _Config()
    config.model_spec = model_spec
    config.hf_assets_path = args.model_path
    config.num_steps = args.num_steps
    config.num_episodes_per_step = args.num_episodes_per_step
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
