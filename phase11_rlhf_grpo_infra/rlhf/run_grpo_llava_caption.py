"""Multimodal GRPO RLHF on LLaVA-Pretrain captions, SGLang rollout.

Mirrors `torchtitan/experiments/rl/simple_grpo_sum_digits.py` with two
production-grade swaps:

1. **Generator engine**: ``SGLangGenerator`` instead of
   ``VLLMGenerator``. Same ``Episode`` shape, same ``group_id``
   convention, so GRPO logic is unchanged. SGLang gives us the
   sequence-dim TP shard + RS+AG fusion fabric (proven in
   ``phase11_rlhf_grpo_infra/PHASE11_SGLANG_REPORT.md``) which produces a different
   NCCL trace pattern than vLLM — the comparison is itself a
   research output.

2. **Task**: ``LlavaCaptionTask`` (multimodal) instead of
   ``SumDigitsTask`` (text-only). Each prompt carries an image_path;
   the generator forwards via ``images=`` to SGLang's multimodal
   inference path. Reward = BLEU-1 vs gold caption + length sanity
   + format bonus (verifiable, no separate reward model).

Usage:

    NCCL_DEBUG=INFO NCCL_DEBUG_FILE=phase11_rlhf_grpo_infra/rlhf/trace/nccl-rank-%h-%p.log \\
    NCCL_DEBUG_SUBSYS=COLL \\
    python phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_caption.py \\
        --model-path /root/torchtitan_attention_residual/phase11_rlhf_grpo_infra/hf/lm_base \\
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
``phase11_rlhf_grpo_infra/rlhf/README.md``. This entry-point is structurally
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

# Make the in-tree task module + torchtitan package importable without installing.
# Bypass any PYTHONPATH/cwd namespace-package ambiguity by inserting the explicit
# torchtitan repo path BEFORE any torchtitan import.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parent.parent  # phase11_rlhf_grpo_infra/rlhf/ → repo root
_TORCHTITAN_DIR = _REPO_ROOT / "torchtitan"
if (_TORCHTITAN_DIR / "torchtitan" / "__init__.py").is_file():
    sys.path.insert(0, str(_TORCHTITAN_DIR))
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torchstore as ts
# ── torchstore Controller monkeypatch ────────────────────────────────────
# Monarch's actor_mesh requires all @endpoint methods on a class to be
# *consistently* async or sync. torchstore 0.1.2 Controller mixes 2 async
# (init, teardown) with 5 sync (get_controller_strategy, locate_volumes,
# notify_put, keys, notify_delete). This crashes with
#     ValueError: <class 'torchstore.controller.Controller'> mixes both
#     async and sync endpoints.
# at actor instantiation. Promote the 5 sync ones to async wrappers
# (they don't actually do async work — Monarch only checks the signature).
# This is the same workaround recipe.json from trace_grpo_kimi_attnres_
# 20260515T074648 referenced as "torchstore Controller monkeypatch —
# promote 5 sync @endpoint to async". It went missing when the file
# was edited later; restored here.
try:
    from torchstore.controller import Controller as _TSController
    from monarch.actor import endpoint as _ts_endpoint
    for _name in ("get_controller_strategy", "locate_volumes",
                  "notify_put", "keys", "notify_delete"):
        _ep = _TSController.__dict__.get(_name)
        if _ep is None:
            continue
        # EndpointProperty wraps the underlying fn in `._method`.
        _src_func = getattr(_ep, "_method", None) or getattr(_ep, "_func", None) or _ep
        if asyncio.iscoroutinefunction(_src_func):
            continue
        def _make_async(fn):
            async def _async_wrapper(self, *a, **kw):
                return fn(self, *a, **kw)
            _async_wrapper.__name__ = fn.__name__
            _async_wrapper.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
            return _async_wrapper
        setattr(_TSController, _name, _ts_endpoint(_make_async(_src_func)))
except Exception as _e:
    import warnings as _w
    _w.warn(f"torchstore Controller monkeypatch failed: {_e}")
# ─────────────────────────────────────────────────────────────────────────

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
            # Belt-and-suspenders: ensure the torchtitan package dir is
            # on PYTHONPATH for the spawned monarch actor, even if the
            # parent's sys.path snapshot missed it (e.g. when launched
            # without PYTHONPATH set in the parent shell).
            _hard_paths = [
                "/workspace/torchtitan_attention_residual",
                "/workspace/torchtitan_attention_residual/torchtitan",
            ]
            for p in _hard_paths:
                if p not in new_pp_parts and os.path.isdir(p):
                    new_pp_parts.insert(0, p)
            for p in existing_pp.split(os.pathsep):
                if p and p not in new_pp_parts:
                    new_pp_parts.append(p)
            os.environ["PYTHONPATH"] = os.pathsep.join(new_pp_parts)
            # Mirror into sys.path so imports below take effect immediately
            # in this worker (before subprocess respawn for monarch).
            import sys as _sys
            for p in reversed(_hard_paths):
                if p not in _sys.path:
                    _sys.path.insert(0, p)
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
            # Apply torchstore Controller sync→async monkeypatch in this
            # worker (mirrors what runs in main process at module import).
            # Critical: trainer.push_model_state_dict instantiates the
            # Controller inside an actor that doesn't import the main
            # script, so the patch must be re-applied here.
            try:
                from phase11_rlhf_grpo_infra.rlhf import (
                    torchstore_controller_monkeypatch as _tcm,
                )
            except Exception as _e:  # noqa: F841
                import importlib.util as _iu
                _spec = _iu.spec_from_file_location(
                    "torchstore_controller_monkeypatch",
                    "/workspace/torchtitan_attention_residual/"
                    "phase11_rlhf_grpo_infra/rlhf/"
                    "torchstore_controller_monkeypatch.py",
                )
                if _spec is not None:
                    _m = _iu.module_from_spec(_spec)
                    _spec.loader.exec_module(_m)

        return _bootstrap


def _make_grader_bootstrap() -> Callable[[], None]:
    """Build a sys.path/PYTHONPATH bootstrap for grader_mesh workers.

    Mirrors what ``Provisioner._make_bootstrap`` does for GPU meshes
    (minus the CUDA pinning + heavy torch pre-imports we don't need
    for a CPU rule-based grader), so the grader actor's pickle
    deserialization can resolve:
      * ``llava_caption_task`` (in phase11_rlhf_grpo_infra/rlhf, not on
        site-packages)
      * ``torchtitan.experiments.rl.types.Episode`` (in the local
        torchtitan repo, not on site-packages either if the parent
        runs out of a non-installed checkout)
    Without this, ``grader.score.call(episodes).get()`` aborts with
    ``ModuleNotFoundError: No module named 'torchtitan.experiments'``
    in the actor; the parent then tears down all meshes and the
    trainer's TCPStore peers disconnect, producing the rank-1/2/3
    "recvValue failed" cascade that masks the real root cause.
    """
    captured = [p for p in list(sys.path) if p and os.path.isdir(p)]
    venv = os.environ.get("VIRTUAL_ENV") or sys.prefix

    def _bootstrap() -> None:
        import os as _os
        import sys as _sys
        hard_paths = [
            "/workspace/torchtitan_attention_residual",
            "/workspace/torchtitan_attention_residual/torchtitan",
            "/workspace/torchtitan_attention_residual/phase11_rlhf_grpo_infra/rlhf",
        ]
        for p in reversed(hard_paths):
            if _os.path.isdir(p) and p not in _sys.path:
                _sys.path.insert(0, p)
        for p in reversed(captured):
            if p and p not in _sys.path:
                _sys.path.insert(0, p)
        existing = _os.environ.get("PYTHONPATH", "")
        merged: list[str] = []
        for p in hard_paths + captured:
            if p and p not in merged and _os.path.isdir(p):
                merged.append(p)
        for p in existing.split(_os.pathsep):
            if p and p not in merged:
                merged.append(p)
        _os.environ["PYTHONPATH"] = _os.pathsep.join(merged)
        _os.environ["VIRTUAL_ENV"] = venv

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
    # Optional torchtitan-native DCP checkpoint directory. When set,
    # the trainer loads weights from here (skipping the HF↔torchtitan
    # state_dict adapter remap entirely). Required for Kimi-Linear
    # AttnRes since no state_dict_adapter ships for it; the safetensors
    # at ``hf_assets_path`` are only consumed by the SGLang generator
    # (which has its own HF loader path), while the trainer needs the
    # DCP that stage 2 SFT wrote. Without this, the trainer runs on
    # random-init weights, immediately overwrites the generator's good
    # weights on step 1 with garbage, and rewards collapse / NaN
    # explodes within a few steps.
    dcp_initial_load_path: str = ""
    num_steps: int = 50
    dump_folder: str = "phase11_rlhf_grpo_infra/rlhf/outputs"
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
    # grader_mesh is CPU-only and would normally be spawned with no
    # bootstrap. But Monarch's spawn_procs() creates fresh Python
    # interpreters whose sys.path is the system default — they do NOT
    # inherit the parent's sys.path additions for phase11_rlhf_grpo_infra/rlhf,
    # the local torchtitan repo, or any editable installs. The grader
    # actor's incoming pickled message references types from those
    # paths (``Episode`` from ``torchtitan.experiments.rl.types``,
    # ``task.reward_function`` bound method from ``llava_caption_task``),
    # so unpickling raises ``ModuleNotFoundError: No module named
    # 'torchtitan.experiments'`` and the actor aborts before any user
    # code runs. The trainer mesh then loses its TCPStore peer when
    # the parent process tears down on the supervision error → cascade
    # of "TCPStore recvValue failed" on trainer ranks 1/2/3.
    # Fix: hand-roll a sys.path bootstrap for the grader mesh (mirrors
    # ``run_grpo_llava_kimi.py::_make_sys_path_bootstrap``).
    grader_mesh = this_host().spawn_procs(
        bootstrap=_make_grader_bootstrap(),
    )

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
                   help="HF-format ckpt dir (e.g. phase11_rlhf_grpo_infra/hf/lm_base)")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--text-only", action="store_true",
                   help="Skip image input (text-only smoke; useful before "
                        "the multimodal SGLang model class lands)")
    p.add_argument("--llava-json",
                   default="/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json")
    p.add_argument("--llava-images",
                   default="/workspace/.hf_home/LLaVA-Pretrain")
    p.add_argument("--dcp-initial-load-path", default="",
                   help="Optional torchtitan-native DCP ckpt for trainer "
                        "model init (e.g. phase5_vlm_multimodal_sft/runs/"
                        "stage2_instruct_sft_447m/checkpoint/step-5200). "
                        "Required for Kimi (no state_dict_adapter), "
                        "otherwise trainer runs on random weights.")
    args = p.parse_args()

    # ModelSpec selection — Kimi-Linear 447M AttnRes (the model we actually
    # trained via stage 2 SFT). PolicyTrainer's earlier hard assert on
    # VarlenAttention was relaxed to a soft warning (torchtitan/experiments/
    # rl/actors/trainer.py:252-277), so non-Qwen3 specs build + parallelize
    # fine. The 2026-05-15 trace_grpo_kimi_attnres run confirmed Kimi runs
    # 60+ steps cleanly; the Qwen3-fallback that briefly lived here was
    # based on the older hard-assert that no longer exists.
    flavor = os.environ.get("RLHF_FLAVOR",
                            "kimi_linear_447m_aligned_block_attn_res_n4")
    if flavor.startswith("kimi"):
        from torchtitan.experiments.kimi_linear import (
            config_registry as kimi_registry,
        )
        from torchtitan.experiments.kimi_linear.parallelize import (
            parallelize_kimi_linear as _orig_kl_parallelize,
        )
        from torchtitan.config.configs import (
            TrainingConfig, ActivationCheckpointConfig,
        )
        from torchtitan.protocols.model_converter import ModelConvertersContainer
        spec_builder = getattr(kimi_registry, flavor, None)
        if spec_builder is None:
            raise ValueError(f"Unknown kimi flavor: {flavor}")
        trainer_cfg = spec_builder()
        model_spec = trainer_cfg.model_spec
        # Wrap parallelize: PolicyTrainer's RL path only passes (model,
        # parallel_dims, parallelism, compile_config). The original
        # parallelize_kimi_linear also needs training/model_converters/
        # ac_config/dump_folder for SFT-time wiring. For RL inference-side
        # forwards, sensible no-op defaults suffice.
        def _kl_rl_parallelize(model, *, parallel_dims, parallelism,
                               compile_config=None, **_kw):
            return _orig_kl_parallelize(
                model,
                parallel_dims=parallel_dims,
                parallelism=parallelism,
                compile_config=compile_config,
                training=TrainingConfig(),
                model_converters=ModelConvertersContainer.Config(),
                ac_config=ActivationCheckpointConfig(),
                dump_folder="/tmp/_rl_kimi_parallelize_no_dump",
            )
        model_spec.parallelize_fn = _kl_rl_parallelize
    else:
        # Optional: keep Qwen3 path available via RLHF_FLAVOR override.
        from torchtitan.models.qwen3 import model_registry as qwen3_registry
        from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
        model_spec = qwen3_registry(flavor)
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
    config.dcp_initial_load_path = args.dcp_initial_load_path
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
