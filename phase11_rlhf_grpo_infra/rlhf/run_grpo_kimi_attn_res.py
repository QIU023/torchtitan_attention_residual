"""GRPO on the 447M Kimi AttnRes LM (real research weights, text-only).

Mirror of run_grpo_sum_digits.py with Qwen3 → Kimi AttnRes swap. The
goal: prove the PolicyTrainer + SGLangGenerator framework works on
our actual research model (1.4B-total / 447M-active Kimi Linear AttnRes,
KDA + MLA + MoE), not a Qwen3 placeholder.

Differences from run_grpo_sum_digits.py:

  * ``model_spec`` from torchtitan/experiments/kimi_linear (has its
    own ``parallelize_kimi_linear`` that handles FSDP+TP correctly,
    not the rl-specific Qwen3 substitute).
  * ``state_dict_adapter=None`` on the model_spec → trainer loads
    via the new ``dcp_initial_load_path`` path (DCP-native, skips
    HF↔torchtitan key remap that doesn't yet exist for Kimi MoE).
  * ``hf_assets_path`` still points to a converted HF safetensors
    dir for the SGLang Engine (which loads HF format by default).
  * No PolicyTrainer Qwen3 inner-attention assertion (already soft-
    warned in upstream branch).

Known caveats (will be flagged at run time):

  * Disk-based weight sync from trainer→generator currently writes
    nothing to ``weight_sync_disk_path``: the trainer's
    ``push_model_state_dict`` only goes through torchstore, not disk.
    Generator's ``update_weights_from_disk`` will see an empty dir
    and skip with a warning — which means policy rollouts use the
    initial v0 weights the entire run. That's still useful for
    end-to-end smoke (does the trainer step, does the SGLang model
    class accept our 447M MoE ckpt, do gradients flow?), but it's
    NOT yet "true PPO with synced weights". The disk dump is
    documented as a follow-up RFC.
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
from torchtitan.experiments.rl.sum_digits import extract_answer, SumDigitsTask
from torchtitan.experiments.rl.types import Episode

# Reuse the Provisioner from the LLaVA entry-point.
from run_grpo_llava_caption import Provisioner  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class _Config(Configurable.Config):
    model_spec: object = None
    hf_assets_path: str = ""
    dcp_initial_load_path: str = ""
    num_steps: int = 50
    dump_folder: str = "phase11_rlhf_grpo_infra/rlhf/outputs/grpo_kimi_attn_res"
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

    # SHM transport requires high ulimit -l (locked-memory); container is
    # only 64KB. Force MonarchRPC instead — slower but works without
    # cudaHostRegister pinning. To enable SHM, restart the container with
    # --ulimit memlock=-1 and switch back to TransportType.Unset.
    from torchstore.transport import TransportType
    await ts.initialize(
        mesh=trainer_mesh,
        strategy=ts.LocalRankStrategy(
            default_transport_type=TransportType.MonarchRPC,
        ),
    )
    trainer.push_model_state_dict.call().get()
    generator.pull_model_state_dict.call(0).get()

    for step in range(config.num_steps):
        t0 = time.perf_counter()

        records = []
        for _ in range(config.num_episodes_per_step):
            q, ans = task.create_question()
            records.append((q, ans))
        prompts = [
            f"{task.get_system_prompt()}\n\nUser: {q}\nAssistant:"
            for q, _ in records
        ]
        gold = [a for _, a in records]

        result = generator.generate.call(prompts, gold, None).get()
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
        help="torchtitan-native DCP ckpt dir (e.g. phase4_kimi_attnres_lm_pretrain/.../step-12500)",
    )
    p.add_argument(
        "--hf-model-path", required=True,
        help="HF safetensors dir for the SGLang Engine to load. "
             "Use phase10_ckpt_dcp_to_hf/dcp_to_hf_kimi_attn_res.py to convert from DCP.",
    )
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--num-episodes-per-step", type=int, default=4)
    p.add_argument(
        "--kl-coef", type=float, default=0.0,
        help="0 = vanilla GRPO; >0 engages frozen ref + KL penalty (PPO mode)",
    )
    p.add_argument(
        "--flavor", default="kimi_linear_447m_aligned_block_attn_res_n4",
        help="torchtitan flavor name from kimi_linear/config_registry.py",
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

    # The rl PolicyTrainer calls parallelize_fn with only
    # (model, parallel_dims, parallelism, compile_config) — but
    # parallelize_kimi_linear requires four more kwargs (training,
    # model_converters, ac_config, dump_folder). Adapt the signature
    # locally so the trainer can drive it without touching core.
    _orig_parallelize = parallelize_kimi_linear
    _adapter_dump_dir = "phase11_rlhf_grpo_infra/rlhf/outputs/grpo_kimi_attn_res"

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
        f"Loaded ModelSpec: name={model_spec.name} flavor={model_spec.flavor}"
    )
    if model_spec.state_dict_adapter is None:
        logger.info(
            "model_spec.state_dict_adapter is None — using DCP-native load via "
            f"--dcp-load-path={args.dcp_load_path}"
        )

    config = _Config()
    config.model_spec = model_spec
    config.hf_assets_path = args.hf_model_path  # SGLang side (HF format)
    config.dcp_initial_load_path = args.dcp_load_path  # trainer side (DCP)
    config.num_steps = args.num_steps
    config.num_episodes_per_step = args.num_episodes_per_step
    config.kl_coef = args.kl_coef

    if args.kl_coef > 0:
        logger.info(f"PPO mode (kl_coef={args.kl_coef}, frozen ref engaged)")
    else:
        logger.info("GRPO mode (kl_coef=0, no frozen ref)")

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
