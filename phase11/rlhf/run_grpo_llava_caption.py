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
    """Allocates GPU ranges for Monarch proc meshes.

    Two allocation modes:

    * ``allocate(n)`` — split mode. Each spawned worker sees a single
      GPU. Used by trainer mesh where each actor is one FSDP rank.
    * ``allocate_shared(n)`` — shared mode. ALL workers in the mesh
      see the same N GPUs. Used by SGLang generator mesh where the
      Engine internally TP-shards onto all visible GPUs and only
      one (lead) actor instance actually constructs it.

    The bootstrap callback runs inside each spawned worker subprocess
    BEFORE CUDA initializes and BEFORE any torch import. We use it to:

    1. Pin ``CUDA_VISIBLE_DEVICES`` so the worker only sees its
       allocated GPU(s).
    2. Re-establish the parent process's venv. Without this, Monarch's
       spawned worker inherits a partially-broken sys.path and may
       import the system torch.

    The env-injection is a no-op outside venv-based deployments.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0
        # Capture the parent process's effective sys.path so the
        # spawned worker subprocesses can re-establish it. Monarch's
        # default spawn doesn't preserve all of: editable installs,
        # venv site-packages, and additional paths added by .pth
        # files. We snapshot what the parent ACTUALLY sees and
        # propagate that via PYTHONPATH.
        self._captured_paths = list(sys.path)
        # Parent's venv (matches activate-style behaviour).
        self._venv = os.environ.get("VIRTUAL_ENV") or sys.prefix

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        """Split mode: ``num_gpus`` workers, each with one GPU."""
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus
        return self._make_bootstrap(gpu_ids, share_all=False)

    def allocate_shared(self, num_gpus: int) -> tuple[Callable[[], None], list[int]]:
        """Shared mode: bootstrap binds every worker in the mesh to
        the same ``num_gpus`` GPUs. Returns (bootstrap_fn, gpu_ids).

        Used for SGLang generator: the Engine inside the lead actor
        spawns its own TP-N workers and needs all N GPUs visible.
        Non-lead actors (rank > 0) are no-ops and don't allocate any
        GPU memory; they exist only because Monarch's
        ``per_host={"gpus": N}`` spawn creates N processes.
        """
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus
        return self._make_bootstrap(gpu_ids, share_all=True), gpu_ids

    def _make_bootstrap(self, gpu_ids: list[int], *, share_all: bool) -> Callable[[], None]:
        # Capture by value for the closure — these strings live in
        # the bootstrap callback that the worker subprocess runs.
        venv = self._venv
        # Filter to filesystem paths the worker can actually use.
        propagated_paths = [
            p for p in self._captured_paths
            if p and os.path.isdir(p)
        ]

        # In split mode each worker gets its own GPU id (1 per actor).
        # In shared mode every worker sees the same gpu_ids list, so
        # SGLang's lead actor can spawn TP=N inner workers across all
        # the GPUs while non-lead actors run no-op.
        cvd_str = ",".join(str(g) for g in gpu_ids)

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd_str
            os.environ["VIRTUAL_ENV"] = venv
            existing_pp = os.environ.get("PYTHONPATH", "")
            new_pp_parts = list(propagated_paths)
            for p in existing_pp.split(os.pathsep):
                if p and p not in new_pp_parts:
                    new_pp_parts.append(p)
            os.environ["PYTHONPATH"] = os.pathsep.join(new_pp_parts)
            venv_bin = f"{venv}/bin"
            existing_path = os.environ.get("PATH", "")
            if venv_bin not in existing_path.split(os.pathsep):
                os.environ["PATH"] = (
                    venv_bin + (os.pathsep + existing_path if existing_path else "")
                )
            # Pre-import all the heavy modules we know will be
            # triggered during pickle deserialize of actor messages.
            # If left for the message handler to import lazily, two
            # bugs hit:
            #   * ``torch._C is not a package`` — torch.distributed.rpc
            #     forces ``torch._C._distributed_c10d`` to resolve as
            #     a submodule; in a fresh interpreter mid-pickle that
            #     fails because ``_C`` is a ``.so`` (not a package).
            #   * ``_DeadlockError`` — the nested import chain
            #     re-enters ``torch.distributed._shard._utils`` while
            #     another thread holds its module lock.
            # Both are eliminated by completing torch / torchtitan /
            # sglang / torchstore imports here, BEFORE Monarch
            # deserializes the first actor message.
            import site
            site.main()
            import torch  # noqa: F401
            import torch.distributed  # noqa: F401
            import torch.distributed.rpc  # noqa: F401
            import torch.distributed._shard._utils  # noqa: F401
            import torch.distributed.checkpoint  # noqa: F401
            import torch.distributed.fsdp  # noqa: F401
            # torchtitan imports (the message handler will trigger
            # these via the model_spec we pass to the actor spawn).
            import torchtitan  # noqa: F401
            import torchtitan.config  # noqa: F401
            import torchtitan.protocols.model_spec  # noqa: F401
            import torchtitan.experiments.rl  # noqa: F401
            import torchtitan.experiments.rl.plugin  # noqa: F401
            # torchstore + monarch (the runtime needs these too,
            # but they may already be loaded by Monarch's bootstrap).
            import torchstore  # noqa: F401

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

    # Provision GPU meshes — 4 GPUs trainer (split, one per actor),
    # 4 GPUs generator (shared, all visible to the lead SGLang Engine).
    provisioner = Provisioner(total_gpus=8)
    trainer_bootstrap = provisioner.allocate(4)
    generator_bootstrap, generator_gpu_ids = provisioner.allocate_shared(4)
    logger.info(f"Generator mesh shares GPUs: {generator_gpu_ids}")

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

        # 2. Rollout. ``call`` returns a ValueMesh — one entry per
        # actor in the mesh. Under our lead/follower pattern only
        # rank-0 actually generated; flatten to its list.
        result = generator.generate.call(prompts, gold, images).get()
        episodes = []
        for _, eps in result:
            episodes.extend(eps)

        # 3. Score. ``call`` returns a ValueMesh (one entry per
        # actor); the CPU grader mesh has 1 actor, take its result.
        result = grader.score.call(episodes).get()
        if isinstance(result, list):
            episodes = result
        else:
            try:
                episodes = next(iter(result))[1]
            except (TypeError, StopIteration):
                # Direct scalar (older Monarch path).
                episodes = result

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
        metrics = trainer.step.call(episodes).get()
        # Take rank-0 metrics from ValueMesh.
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

    # ModelSpec selection.
    #
    # Architectural note: torchtitan's ``PolicyTrainer`` is currently
    # Qwen3-specific. Its ``_build_model`` does:
    #
    #     assert isinstance(
    #         model_spec.model.layers[0].attention.inner_attention,
    #         VarlenAttention.Config,
    #     )
    #
    # which assumes Qwen3-style ``.layers[0].attention.inner_attention``
    # access at config time. Our Kimi Linear AttnRes (KDA + MLA) has a
    # different structure (no ``.attention.inner_attention`` field) so
    # PolicyTrainer can't load our 447m AttnRes ckpt as-is.
    #
    # Filing this as a separate upstream RFC ("make PolicyTrainer
    # model-agnostic"). Until that lands, we use Qwen3-0.6B for the
    # framework + NCCL-trace deliverable. The SGLangGenerator and the
    # multimodal task pieces are unchanged — when PolicyTrainer
    # supports a non-Qwen3 model_spec, swapping the trainer side is
    # a one-config edit.
    from torchtitan.models.qwen3 import model_registry as qwen3_registry
    # 0.6B (non-varlen) avoids the torch 2.9 ``varlen_attn`` stub
    # raise. Upstream defaults to ``0.6B_varlen`` which requires
    # torch ≥2.10 nightly. ``0.6B`` uses standard flash attention
    # which is bundled with torch 2.9 stable.
    flavor = os.environ.get("RLHF_FLAVOR", "0.6B")
    model_spec = qwen3_registry(flavor)
    # Mirror upstream simple_grpo_sum_digits.py: swap in the RL-side
    # parallelize fn (the one in models/qwen3 expects training-side
    # kwargs that PolicyTrainer doesn't supply at RL time).
    from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
    model_spec.parallelize_fn = parallelize_qwen3
    logger.info(f"Loaded ModelSpec: name={model_spec.name} flavor={model_spec.flavor}")
    logger.info(
        "NOTE: PolicyTrainer is Qwen3-specific upstream — using Qwen3 for "
        "the framework + trace; the 447m Kimi AttnRes needs a "
        "model-agnostic PolicyTrainer (separate RFC)."
    )

    config = _Config()
    config.model_spec = model_spec
    config.hf_assets_path = args.model_path
    config.num_steps = args.num_steps
    config.text_only = args.text_only
    config.llava_json_path = args.llava_json
    config.llava_images_dir = args.llava_images

    # Trainer + generator parallelism are inherited from default configs.
    # Override here for our 8x 5090 layout: trainer FSDP=4, generator TP=4.
    config.trainer.parallelism.data_parallel_shard_degree = 4
    config.generator.parallelism.tensor_parallel_degree = 4
    # SGLang's mem-pool calc is:
    #   rest = post_model_load_mem - pre_model_load_mem * (1 - mem_frac)
    # So HIGHER mem_frac = less of pre_load_mem subtracted = bigger
    # KV pool. Counter-intuitive vs vLLM where higher mem_frac means
    # "use more of my total GPU budget". Push to 0.85 — at that
    # value the formula leaves enough head-room for the KV cache.
    config.generator.gpu_memory_limit = 0.85
    config.generator.weight_sync_method = "disk"
    config.generator.weight_sync_disk_path = (
        config.dump_folder + "/sglang_weights"
    )
    Path(config.generator.weight_sync_disk_path).mkdir(parents=True, exist_ok=True)

    asyncio.run(_async_main(config))


if __name__ == "__main__":
    main()
