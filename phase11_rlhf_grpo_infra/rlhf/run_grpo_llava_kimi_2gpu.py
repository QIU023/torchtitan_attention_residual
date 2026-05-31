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
    python phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \\
        --dcp-load-path $PWD/phase5_vlm_multimodal_sft/runs/vlm_447m_sft_instruct/checkpoint/step-2344 \\
        --hf-model-path $PWD/phase11_rlhf_grpo_infra/hf/vlm_sft_1ep \\
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

# Exclude AttnRes pseudo-query projections (Linear D->1) from fp8 quant.
# attn_res.py:128 docstring + empirical fp8 phase-1 einsum cuBLAS crash
# on Blackwell. Set before SGLang Engine boots in the generator subprocess.
# Harmless when fp8 quant isn't used.
os.environ.setdefault(
    "SGLANG_FP8_IGNORED_LAYERS",
    "attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts",
)

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

# Pre-import sglang overlays so AutoConfig.register('kimi_attn_res_vl', ...)
# fires before SGLang generator tries to load the HF config of our VLM ckpt.
# Without this the generator dies with `KeyError: 'kimi_attn_res_vl'` in
# transformers.AutoConfig.from_pretrained. Idempotent — safe even if the
# overlays are missing in non-VLM setups (caught by ImportError).
try:
    import sglang.srt.configs.kimi_attn_res_vl  # noqa: F401
    import sglang.srt.models.attn_res_vl_overlay  # noqa: F401
except ImportError as _e:
    pass

# Patch sglang ModelRunner.kimi_linear_config to also recognize
# KimiAttnResVLConfig (which holds a KimiLinearConfig in text_config).
# Without this, the runner doesn't classify the VLM as "mambaish",
# fails to attach KDAAttnBackend/HybridLinearAttnBackend, and the
# linear-attn path in radix_linear_attention.py calls the main
# flashinfer/triton backend with mixed_qkv/a/b kwargs → TypeError
# "AttentionBackend.forward() missing 3 required positional arguments:
# 'q', 'k', and 'v'". This is the root cause of yesterday's GRPO crash.
try:
    from sglang.srt.configs.kimi_linear import KimiLinearConfig
    from sglang.srt.configs.kimi_attn_res_vl import KimiAttnResVLConfig
    from sglang.srt.model_executor import model_runner as _mr_mod

    def _kimi_linear_config_patched(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        # VLM wrapper: text_config holds the kimi_linear LM config.
        if isinstance(config, KimiAttnResVLConfig):
            inner = getattr(config, "text_config", None)
            if isinstance(inner, KimiLinearConfig):
                return inner
        return None

    _mr_mod.ModelRunner.kimi_linear_config = property(_kimi_linear_config_patched)
except ImportError as _e:
    pass

# Patch torchstore 0.1.2's Controller class so monarch will spawn it.
# monarch's actor_mesh.py:1587 raises ValueError when an Actor mixes
# sync and async @endpoint methods. Controller defines 5 sync endpoints
# (get_controller_strategy, locate_volumes, notify_put, keys,
# notify_delete) and 2 async (init, teardown). Promote the sync ones to
# async coroutine functions — bodies don't await, so the wrap is
# semantically a no-op. Done in-file (NOT a version pin or
# site-packages edit) so the sglang/monarch/torchstore environment
# versions are unchanged.
#
# CRITICAL: this patch must run in EVERY process that instantiates
# Controller. The main process is patched at import time. Spawned
# trainer/generator subprocesses get a fresh interpreter, so their
# bootstrap callback re-applies the patch (see _bootstrap_with_torchstore_patch).
def _monarch_major_minor():
    """(major, minor) of installed torchmonarch, or None if undetectable."""
    try:
        from importlib.metadata import version
        parts = version("torchmonarch").split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def _patch_torchstore_controller():
    # VERSION GUARD (2026-05-31): the asyncify wrap below is ONLY needed for
    # monarch 0.1.x, whose actor_mesh raised ValueError when an Actor mixed
    # sync + async @endpoint methods. monarch >=0.2 (we run 0.5.0 on torch
    # 2.11) dispatches per-method via inspect.iscoroutinefunction in its
    # _method_cache and NATIVELY supports mixed sync/async endpoints — so
    # asyncify-ing torchstore's sync Controller endpoints there is not just
    # unnecessary but HARMFUL: it turns sync endpoints into coroutines that
    # torchstore 0.1.2 calls synchronously, hanging get_state_dict forever.
    # Old torch-2.9 + monarch-0.1.2 envs still hit the original code path.
    mm = _monarch_major_minor()
    if mm is not None and mm >= (0, 2):
        return  # monarch >=0.2: no patch; native mixed sync/async dispatch
    try:
        import functools
        import inspect
        from torchstore.controller import Controller

        sync_names = (
            "get_controller_strategy",
            "locate_volumes",
            "notify_put",
            "keys",
            "notify_delete",
        )

        def _asyncify(fn):
            if inspect.iscoroutinefunction(fn):
                return fn

            @functools.wraps(fn)
            async def _async_wrapper(*args, _fn=fn, **kwargs):
                return _fn(*args, **kwargs)

            return _async_wrapper

        for name in sync_names:
            ep = getattr(Controller, name, None)
            if ep is None:
                continue
            method = getattr(ep, "_method", None)
            if method is None or inspect.iscoroutinefunction(method):
                continue
            ep._method = _asyncify(method)
    except (ImportError, AttributeError):
        pass


def _bootstrap_with_torchstore_patch(orig_bootstrap):
    """Wrap a Provisioner bootstrap callable so spawned subprocesses
    apply the torchstore Controller patch before any actor instantiation.
    """
    def _wrapped():
        _patch_torchstore_controller()
        orig_bootstrap()

    return _wrapped


def _make_sys_path_bootstrap():
    """Bootstrap that re-establishes the parent's sys.path inside a
    spawned subprocess. Provisioner already does this for GPU-allocating
    meshes; this is the no-GPU analogue used for grader_mesh, whose
    actors otherwise can't deserialize args referencing modules outside
    standard site-packages (e.g. ``llava_caption_task`` in phase11_rlhf_grpo_infra/rlhf,
    referenced via the pickled ``task.reward_function`` bound method).
    """
    import os
    import sys
    captured = [p for p in list(sys.path) if p and os.path.isdir(p)]

    def _bootstrap():
        import os as _os
        import sys as _sys
        for p in reversed(captured):
            if p and p not in _sys.path:
                _sys.path.insert(0, p)
        existing = _os.environ.get("PYTHONPATH", "")
        new = ":".join(captured)
        _os.environ["PYTHONPATH"] = (new + ":" + existing) if existing else new

    return _bootstrap


# Apply to the main process at import time.
_patch_torchstore_controller()

from torchtitan.config import Configurable
from torchtitan.experiments.rl.actors.grader import Grader
from torchtitan.experiments.rl.actors.sglang_generator import SGLangGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.types import Episode

from llava_caption_task import LlavaCaptionTask  # noqa: E402
from gqa_vqa_task import GqaVqaTask  # noqa: E402
from llava_opd_task import LlavaOpdTask  # noqa: E402
from vqa_aligned_opd_task import VqaAlignedOpdTask  # noqa: E402
from run_grpo_llava_caption import Provisioner  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class _Config(Configurable.Config):
    model_spec: object = None
    hf_assets_path: str = ""
    dcp_initial_load_path: str = ""
    num_steps: int = 500
    dump_folder: str = "phase11_rlhf_grpo_infra/rlhf/outputs/grpo_llava_kimi"
    num_episodes_per_step: int = 4
    log_samples: bool = True
    kl_coef: float = 0.0
    llava_json_path: str = "/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
    llava_images_dir: str = "/workspace/.hf_home/LLaVA-Pretrain"
    task: str = "llava"  # "llava" (GRPO caption/BLEU), "gqa" (GRPO VQA exact-match), "opd" (on-policy distillation)

    # OPD-only fields (ignored when task != "opd").
    teacher_model_id: str = "llava-hf/llama3-llava-next-8b-hf"
    teacher_device: str = "cuda:0"  # rides on the trainer GPU (single-rank OPD)
    tokenizer_path: str = ""  # defaults to hf_assets_path when empty
    # GKD loss hyperparameters (Agarwal et al. 2024). β=0.5 / T=1.0
    # match TRL's generalized_jsd_loss defaults (symmetric JSD).
    opd_beta: float = 0.5
    opd_temperature: float = 1.0
    # Ckpt every N OPD steps. 0 disables. Saves to
    # ``{dump_folder}/opd_ckpts/step-N/`` as DCP (no optim state).
    opd_ckpt_interval: int = 0
    opd_ckpt_dir: str = ""  # defaults to dump_folder + "/opd_ckpts" when empty
    opd_task_type: str = "caption"  # "caption" (LlavaOpdTask) or "vqa" (VqaAlignedOpdTask)

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


async def _async_main_opd(config: _Config) -> None:
    """OPD (on-policy distillation) main loop.

    GPU layout (no idle cards):
      * Trainer rank 0: physical cuda:0 (logical cuda:0) — student only.
      * Generator TP=4: physical cuda:1-4 (allocate_shared at next_gpu=1).
      * Teacher device_map: physical cuda:5-7 (logical cuda:1-3 inside
        trainer process after CVD expansion).
      * No idle GPUs.

    Differences from GRPO main loop:
      * ``LauncherOPDTrainer`` (HF teacher inside actor), not
        ``PolicyTrainer``.
      * Single trainer rank (``data_parallel_shard_degree=1``) so we
        avoid cross-rank teacher coordination.
      * Trainer bootstrap is wrapped to expand ``CUDA_VISIBLE_DEVICES``
        from ``"0"`` to ``"0,5,6,7"`` so HF accelerate can place the
        teacher on the otherwise-idle cuda:5-7.
      * No grader actor and no advantage computation. The teacher's
        logits are the entire supervision signal.
    """
    from opd_trainer_launcher import LauncherOPDTrainer

    if config.opd_task_type == "vqa":
        task = VqaAlignedOpdTask(
            json_path=config.llava_json_path,
            images_dir=config.llava_images_dir,
        )
    else:
        task = LlavaOpdTask(
            json_path=config.llava_json_path,
            images_dir=config.llava_images_dir,
        )
    logger.info(
        f"OPD task ({config.opd_task_type}) loaded: {len(task)} records "
        f"from {config.llava_json_path}"
    )

    # Allocate trainer=1 (cuda:0) + generator_shared=4 (cuda:1-4). cuda:5-7
    # are left unallocated by Provisioner; the trainer bootstrap wrapper
    # below will expose them to the trainer process for HF teacher load.
    TEACHER_PHYS_GPUS = []
    provisioner = Provisioner(total_gpus=2)
    trainer_bootstrap_base = provisioner.allocate(1)
    generator_bootstrap, gen_gpu_ids = provisioner.allocate_shared(1)
    # Bump Provisioner's next_gpu past the teacher cards so no later
    # allocation can grab them (defensive — there's no later allocation
    # in this path, but stays robust if someone adds one).
    provisioner.next_gpu = 2

    def _expand_cvd_for_teacher(base_bs):
        """Wrap the trainer bootstrap to (a) extend CUDA_VISIBLE_DEVICES
        with the teacher cards, and (b) inject the rlhf/ folder into
        sys.path so the spawned trainer subprocess can unpickle
        ``LauncherOPDTrainer`` (which lives in rlhf/, not in any
        site-packages path that Provisioner propagates).

        Inside the trainer process, ``cuda:0`` still maps to physical
        GPU 0 (the student card); physical cuda:5,6,7 appear as logical
        cuda:1,2,3 — what HF's accelerate sees when we pass
        ``max_memory={1: ..., 2: ..., 3: ...}``.
        """
        rlhf_dir = os.path.dirname(os.path.abspath(__file__))
        def _wrapped():
            base_bs()
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            extra = ",".join(str(g) for g in TEACHER_PHYS_GPUS)
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{cvd},{extra}"
            # rlhf folder onto sys.path AND PYTHONPATH — base_bs has
            # already set PYTHONPATH but doesn't include the launcher's
            # own folder. cloudpickle.loads needs ``import
            # opd_trainer_launcher`` to resolve.
            import sys as _sys
            if rlhf_dir not in _sys.path:
                _sys.path.insert(0, rlhf_dir)
            existing_pp = os.environ.get("PYTHONPATH", "")
            if rlhf_dir not in existing_pp.split(os.pathsep):
                os.environ["PYTHONPATH"] = (
                    rlhf_dir + os.pathsep + existing_pp
                    if existing_pp else rlhf_dir
                )
        return _wrapped

    trainer_bootstrap = _expand_cvd_for_teacher(trainer_bootstrap_base)
    trainer_bootstrap = _bootstrap_with_torchstore_patch(trainer_bootstrap)
    generator_bootstrap = _bootstrap_with_torchstore_patch(generator_bootstrap)
    logger.info(
        f"OPD layout: trainer cuda:0 (+teacher dev_map on phys cuda:{TEACHER_PHYS_GPUS}); "
        f"generator TP=4 on phys cuda:{gen_gpu_ids}; idle=0"
    )

    trainer_mesh = this_host().spawn_procs(
        per_host={"gpus": 1}, bootstrap=trainer_bootstrap,
    )
    generator_mesh = this_host().spawn_procs(
        per_host={"gpus": 1}, bootstrap=generator_bootstrap,
    )

    await setup_torch_elastic_env_async(trainer_mesh)
    await setup_torch_elastic_env_async(generator_mesh)

    trainer = trainer_mesh.spawn(
        "trainer",
        LauncherOPDTrainer,
        config.trainer,
        model_spec=config.model_spec,
        hf_assets_path=config.hf_assets_path,
        transfer_dtype=config.generator.model_dtype,
        kl_coef=0.0,  # OPD: ref-model KL is wasted — teacher KL is the loss
        dcp_initial_load_path=config.dcp_initial_load_path,
    )
    generator = generator_mesh.spawn(
        "generator",
        SGLangGenerator,
        config.generator,
        model_spec=config.model_spec,
        model_path=config.hf_assets_path,
    )

    await ts.initialize(
        mesh=trainer_mesh,
        strategy=ts.LocalRankStrategy(),
    )
    trainer.push_model_state_dict.call().get()
    generator.pull_model_state_dict.call(0).get()

    # Lazy-load HF teacher + tokenizer + loss fn INSIDE the trainer process.
    # The init endpoint takes only strings + a small dict (pickleable).
    opd_loss_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = config.tokenizer_path or config.hf_assets_path
    # Pin teacher to LOGICAL devices 1,2,3 (= physical cuda:5,6,7 after
    # the trainer bootstrap CVD expansion). max_memory ~7 GiB / card for
    # an 8B bf16 model spread across 3 cards leaves ~5 GiB free per card
    # for activations + KV cache during teacher forward.
    teacher_max_memory = {1: "7GiB", 2: "7GiB", 3: "7GiB"}
    init_diag = trainer.init_opd_components.call(
        teacher_model_id=config.teacher_model_id,
        teacher_device=config.teacher_device,  # unused when max_memory set
        tokenizer_path=tokenizer_path,
        opd_loss_module_dir=opd_loss_dir,
        teacher_max_memory=teacher_max_memory,
    ).get()
    logger.info(f"OPD components initialized: {init_diag}")

    # CRITICAL: load vision tower + projector for the student's forward.
    # Without this, compute_response_logits sees image_token_id as a
    # literal text token (no vision embeddings spliced in) → student
    # trains in a different input distribution than it evals in →
    # GQA acc collapses (verified Stage D-2: 12.3% → 0.67%).
    vision_diag = trainer.init_vision_from_hf.call(
        hf_model_path=config.hf_assets_path,
        vision_tower_id="google/siglip-base-patch16-224",
    ).get()
    logger.info(f"OPD vision components loaded: {vision_diag}")

    # Inject GKD hyperparameters (β, temperature) into the trainer actor.
    trainer.set_opd_hyperparams.call(
        beta=config.opd_beta, temperature=config.opd_temperature,
    ).get()

    # Resolve ckpt directory (used by the periodic save below).
    opd_ckpt_dir = config.opd_ckpt_dir or os.path.join(
        config.dump_folder, "opd_ckpts"
    )
    if config.opd_ckpt_interval > 0:
        Path(opd_ckpt_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"OPD ckpt: every {config.opd_ckpt_interval} steps → {opd_ckpt_dir}"
        )

    for step in range(config.num_steps):
        t0 = time.perf_counter()

        records = [
            task.create_question() for _ in range(config.num_episodes_per_step)
        ]
        prompts = [
            f"{task.get_system_prompt()}\n\n<image>\nUser: {r.prompt_text}\nAssistant:"
            for r in records
        ]
        # OPD has no gold answer — pass empty strings to keep generator API stable.
        gold = ["" for _ in records]
        import base64
        images = []
        for r in records:
            with open(r.image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            images.append(f"data:image/jpeg;base64,{b64}")

        result = generator.generate.call(prompts, gold, images).get()
        episodes: list[Episode] = []
        for _, eps in result:
            episodes.extend(eps)

        # Skip grader + advantage entirely — OPDTrainer.step ignores both.
        metrics = trainer.step.call(episodes).get()
        first_metric = None
        try:
            for _, m in metrics:
                first_metric = m
                break
        except TypeError:
            first_metric = metrics
        loss = first_metric.get("loss", 0.0) if isinstance(first_metric, dict) else 0.0
        n_resp = (
            first_metric.get("num_response_tokens", 0)
            if isinstance(first_metric, dict) else 0
        )

        trainer.push_model_state_dict.call().get()
        generator.pull_model_state_dict.call(step + 1).get()

        dt = time.perf_counter() - t0
        logger.info(
            f"step {step:3d}  loss={loss:.4f}  resp_tokens={n_resp}  dt={dt:.1f}s"
        )
        if config.log_samples and step % 5 == 0 and episodes:
            cand = episodes[0].text[:200].replace("\n", " ").strip()
            logger.info(f"       sample: {cand}")

        # Periodic ckpt save. (step+1) so the FIRST save lands at step
        # opd_ckpt_interval-1 (i.e. after ``interval`` real updates),
        # not at step 0.
        if (config.opd_ckpt_interval > 0
                and (step + 1) % config.opd_ckpt_interval == 0):
            saved = trainer.save_dcp.call(
                save_dir=opd_ckpt_dir, step=step + 1,
            ).get()
            logger.info(f"       ckpt saved at step {step+1}: {saved}")


async def _async_main(config: _Config) -> None:
    # OPD path forks early — different trainer (OPDTrainer not PolicyTrainer),
    # different mesh allocation (single trainer rank to colocate teacher),
    # no grader / no advantages. Keep the GRPO path below untouched.
    if config.task == "opd":
        await _async_main_opd(config)
        return

    if config.task == "gqa":
        task = GqaVqaTask(
            json_path=config.llava_json_path,
            images_dir=config.llava_images_dir,
        )
    else:
        task = LlavaCaptionTask(
            json_path=config.llava_json_path,
            images_dir=config.llava_images_dir,
        )
    logger.info(f"Loaded {type(task).__name__} (task={config.task}) with {len(task)} records")

    provisioner = Provisioner(total_gpus=2)
    trainer_bootstrap = provisioner.allocate(1)
    generator_bootstrap, gen_gpu_ids = provisioner.allocate_shared(1)
    # Wrap bootstraps so spawned subprocesses also patch torchstore.Controller
    # (the main-process patch above doesn't propagate through monarch's
    # spawn_procs, which uses fresh interpreters).
    trainer_bootstrap = _bootstrap_with_torchstore_patch(trainer_bootstrap)
    generator_bootstrap = _bootstrap_with_torchstore_patch(generator_bootstrap)
    logger.info(f"Generator mesh shares GPUs: {gen_gpu_ids}")

    trainer_mesh = this_host().spawn_procs(
        per_host={"gpus": 1}, bootstrap=trainer_bootstrap,
    )
    generator_mesh = this_host().spawn_procs(
        per_host={"gpus": 1}, bootstrap=generator_bootstrap,
    )
    # grader_mesh has no Provisioner bootstrap (CPU-only mesh), so spawned
    # subprocesses get a fresh sys.path missing phase11_rlhf_grpo_infra/rlhf — pickle of
    # task.reward_function then can't find ``llava_caption_task``.
    # Hand-rolled sys-path bootstrap + torchstore patch wrapper.
    grader_bootstrap = _bootstrap_with_torchstore_patch(_make_sys_path_bootstrap())
    grader_mesh = this_host().spawn_procs(bootstrap=grader_bootstrap)

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
    p.add_argument("--task", default="llava", choices=["llava", "gqa", "opd"],
                   help="llava=caption/BLEU (degenerate, model trained on it); "
                        "gqa=VQA exact-match (verifiable capability reward); "
                        "opd=on-policy distillation against an HF teacher "
                        "(no reward; teacher logits are the supervision).")
    p.add_argument("--data-json", default="", help="override task json path")
    p.add_argument("--images-dir", default="", help="override task images dir")
    # OPD-only knobs (ignored when --task != opd).
    p.add_argument("--teacher-model-id", default="llava-hf/llama3-llava-next-8b-hf",
                   help="HF model id for the OPD teacher (must share Llama-3 "
                        "base vocab with the student).")
    p.add_argument("--teacher-device", default="cuda:0",
                   help="CUDA device for the OPD teacher (default cuda:0 = "
                        "trainer rank 0; rides on the student's GPU).")
    p.add_argument("--tokenizer-path", default="",
                   help="HF tokenizer dir for prompt/response decode "
                        "(defaults to --hf-model-path when empty).")
    p.add_argument("--opd-beta", type=float, default=0.5,
                   help="GKD β: 0=reverse-KL(student||teacher), "
                        "1=forward-KL(teacher||student) classical KD, "
                        "0.5=symmetric JSD (Agarwal 2024 paper default).")
    p.add_argument("--opd-temperature", type=float, default=1.0,
                   help="Softmax temperature for distillation. "
                        "1.0=no scaling; 2.0=standard dark-knowledge KD.")
    p.add_argument("--opd-ckpt-interval", type=int, default=0,
                   help="Save trainer DCP every N OPD steps (0=disabled).")
    p.add_argument("--opd-ckpt-dir", default="",
                   help="OPD ckpt root (defaults to "
                        "{dump_folder}/opd_ckpts when empty).")
    p.add_argument("--opd-lr", type=float, default=1e-5,
                   help="Learning rate for OPD distillation. torchtitan "
                        "default is 8e-4 (from-scratch pretraining), "
                        "which is 80x too high for continual distillation "
                        "of an already-SFT'd model. DeepSeek-R1-distill "
                        "uses 5e-6; LLaMA-3 distill 1e-5. Default 1e-5.")
    p.add_argument("--opd-weight-decay", type=float, default=0.01,
                   help="Weight decay for OPD. Default 0.01 (LLaVA-SFT "
                        "convention) vs torchtitan default 0.1.")
    p.add_argument("--opd-task-type", default="caption", choices=["caption", "vqa"],
                   help="caption=LlavaOpdTask(fixed 'describe' prompt); "
                        "vqa=VqaAlignedOpdTask(real VQA questions from mix665k "
                        "conversations, task-aligned with GQA eval).")
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

    # The 447m_aligned ckpts were trained with the config_registry "_n4"
    # variant (num_blocks=4), which model_registry does NOT expose — it only
    # resolves 447m_aligned block_attn_res -> num_blocks=8 (full_attn_res->16).
    # Loading the num_blocks=4 ckpt into an 8-block skeleton would run (the
    # attn_res tensors are shape-invariant to num_blocks) but group the AttnRes
    # forward differently than training -> wrong logits / degraded reward. So for
    # "_n4" flavors we source the ModelSpec from config_registry (the exact spec
    # the SFT + the DCP->HF converter used). Non-_n4 flavors keep model_registry.
    if args.flavor.endswith("_n4"):
        from torchtitan.experiments.kimi_linear import config_registry as _cr
        model_spec = getattr(_cr, args.flavor)().model_spec
        print(f"[grpo] flavor '{args.flavor}' -> config_registry ModelSpec (num_blocks=4)")
    else:
        model_spec = kimi_registry(args.flavor)
    _orig_parallelize = parallelize_kimi_linear
    _adapter_dump_dir = "phase11_rlhf_grpo_infra/rlhf/outputs/grpo_llava_kimi"

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
    config.task = args.task
    # Task-default data paths (override with --data-json / --images-dir).
    if args.task == "gqa":
        config.llava_json_path = args.data_json or "/workspace/gqa_rl/gqa_testdev.json"
        config.llava_images_dir = args.images_dir or "/workspace/gqa_rl"
    elif args.task == "opd":
        config.llava_json_path = (
            args.data_json or "/workspace/llava_opd/llava_v1_5_mix665k.json"
        )
        config.llava_images_dir = args.images_dir or "/workspace/llava_opd/images"
        config.teacher_model_id = args.teacher_model_id
        config.teacher_device = args.teacher_device
        config.tokenizer_path = args.tokenizer_path
        config.opd_beta = args.opd_beta
        config.opd_temperature = args.opd_temperature
        config.opd_ckpt_interval = args.opd_ckpt_interval
        config.opd_ckpt_dir = args.opd_ckpt_dir
        config.opd_task_type = args.opd_task_type
        # CRITICAL: override the from-scratch LR. Stage D-2/D-3 ran with
        # torchtitan default lr=8e-4 → student weights drifted 80x past
        # what continual distillation tolerates → GQA collapsed 12.3%→0%.
        config.trainer.optimizer.lr = args.opd_lr
        config.trainer.optimizer.weight_decay = args.opd_weight_decay
        logger.info(
            f"OPD optimizer overrides: lr={args.opd_lr} "
            f"weight_decay={args.opd_weight_decay} (vs from-scratch "
            f"defaults lr=8e-4, wd=0.1)"
        )
    else:
        if args.data_json:
            config.llava_json_path = args.data_json
        if args.images_dir:
            config.llava_images_dir = args.images_dir

    if args.task == "opd":
        # OPD: single trainer rank (teacher rides on cuda:0 next to student);
        # generator on cuda:1-4 with TP=4. cuda:5-7 left idle for headroom.
        config.trainer.parallelism.data_parallel_shard_degree = 1
        config.generator.parallelism.tensor_parallel_degree = 1
    else:
        config.trainer.parallelism.data_parallel_shard_degree = 1
        config.generator.parallelism.tensor_parallel_degree = 1
    config.generator.gpu_memory_limit = 0.85
    # Block AttnRes residual stream grows unboundedly with depth; on
    # Blackwell (SM 12.0) flashinfer_mla bf16-NaNs at the deep MLA
    # layers. ATTNRES_MLA_FP32_FALLBACK=1 (env, below) handles prefill;
    # decode needs eager SDPA. Without this the rollout generator emits
    # all-`!` garbage and reward collapses to -1.0 (the v16 GRPO
    # failure). See phase11_rlhf_grpo_infra/VISION_INJECTION_BUG_RCA.md.
    config.generator.backend.decode_attention_backend = "torch_native"
    # torch_native has no CUDA-graph support — disable graph capture.
    config.generator.compile.cuda_graph = False
    config.generator.weight_sync_method = "disk"
    config.generator.weight_sync_disk_path = (
        config.dump_folder + "/sglang_weights"
    )
    Path(config.generator.weight_sync_disk_path).mkdir(parents=True, exist_ok=True)

    asyncio.run(_async_main(config))


if __name__ == "__main__":
    main()
