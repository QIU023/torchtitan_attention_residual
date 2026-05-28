"""Launcher-side OPDTrainer adapter.

The upstream-bound ``OPDTrainer`` in
``torchtitan/experiments/rl/actors/opd_trainer.py`` is intentionally HF-
free: it takes the teacher / loss fn / tokenizer via inject endpoints
that accept *callables*. That works when the objects are pickleable and
already live in the trainer process — for unit tests, smaller stand-in
teachers, etc.

For our real Stage-C launcher the teacher is a full HF
``LlavaNextForConditionalGeneration`` (8B params, ~16 GB) which:
  * is not safely pickleable across an actor boundary, and
  * needs to be loaded *inside* the trainer's subprocess on its own
    GPU (so weights live next to where the loss + backward happen).

``LauncherOPDTrainer`` is a thin subclass that adds one extra endpoint
(``init_opd_components``) which takes only strings/ints (HF model id,
tokenizer path, device str) and constructs the heavy objects *inside*
the actor. The upstream base trainer doesn't need to know any of this.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import torch
from monarch.actor import endpoint

from torchtitan.experiments.rl.actors.opd_trainer import OPDTrainer

logger = logging.getLogger(__name__)


class LauncherOPDTrainer(OPDTrainer):
    """OPDTrainer that lazy-loads HF teacher + tokenizer + loss fn inside
    the actor process, given only string config.

    All HF / TRL imports happen inside ``init_opd_components`` so the
    main launcher process (which spawns the actor) doesn't pay for them.
    """

    @endpoint
    async def init_opd_components(
        self,
        teacher_model_id: str,
        teacher_device: str,
        tokenizer_path: str,
        opd_loss_module_dir: str,
        teacher_max_memory: dict | None = None,
    ) -> dict:
        """Build heavy components inside this trainer's process.

        Args:
            teacher_model_id: HF id for the teacher (e.g.
                ``llava-hf/llama3-llava-next-8b-hf``).
            teacher_device: CUDA device for the teacher when single-GPU
                (e.g. ``cuda:0``). Ignored when ``teacher_max_memory``
                is provided.
            teacher_max_memory: optional accelerate ``max_memory`` dict
                (``{device_idx: "10GiB", ...}``). When set, HF
                ``device_map="auto"`` spreads teacher layers across
                the keyed devices. Used by the runner to put the
                teacher on otherwise-idle GPUs (typically cuda:5-7
                physical → cuda:1-3 logical after the trainer
                bootstrap remaps ``CUDA_VISIBLE_DEVICES``).
            tokenizer_path: HF model dir whose tokenizer matches the
                student's vocab (used to decode Episode prompt/response
                token ids back to text for the teacher's input).
            opd_loss_module_dir: directory containing ``opd_loss.py``
                (the launcher-side adapter that calls
                ``trl.experimental.gkd.GKDTrainer.generalized_jsd_loss``).
                Added to ``sys.path`` so the import works regardless of
                the actor's working dir.

        Returns:
            Diagnostic dict: ``teacher_vocab``, ``tokenizer_vocab``,
            ``device``.
        """
        # Make the launcher-side opd_loss adapter importable inside
        # this subprocess. The Provisioner bootstrap already added
        # phase11_rlhf_grpo_infra/rlhf to sys.path; this is belt-and-
        # braces for cases where the spawn used a clean interpreter.
        if opd_loss_module_dir and opd_loss_module_dir not in sys.path:
            sys.path.insert(0, opd_loss_module_dir)

        from teacher_scorer import TeacherScorer  # noqa: E402
        from opd_loss import opd_loss  # noqa: E402
        from transformers import AutoTokenizer  # noqa: E402

        if teacher_max_memory is not None:
            logger.info(
                f"OPD init: loading teacher {teacher_model_id} "
                f"device_map=auto max_memory={teacher_max_memory}"
            )
            scorer = TeacherScorer(
                model_id=teacher_model_id,
                dtype=torch.bfloat16,
                max_memory=teacher_max_memory,
            )
        else:
            logger.info(
                f"OPD init: loading teacher {teacher_model_id} on {teacher_device}"
            )
            scorer = TeacherScorer(
                model_id=teacher_model_id,
                device=teacher_device,
                dtype=torch.bfloat16,
            )
        self._teacher_score_fn = scorer.score

        logger.info(f"OPD init: loading tokenizer from {tokenizer_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self._opd_loss_fn = opd_loss

        v_teacher = scorer.model.lm_head.out_features
        v_tok = len(self._tokenizer)
        logger.info(
            f"OPD init done: teacher_vocab={v_teacher}, "
            f"tokenizer_vocab={v_tok}, device={teacher_device}"
        )
        return {
            "teacher_vocab": v_teacher,
            "tokenizer_vocab": v_tok,
            "device": teacher_device,
        }

    @endpoint
    async def init_vision_from_hf(
        self,
        hf_model_path: str,
        vision_tower_id: str = "google/siglip-base-patch16-224",
    ) -> dict:
        """Load SigLIP vision tower + 2-layer projector from the SFT HF dir.

        **Critical for OPD correctness**: ``compute_response_logits``
        injects ``vision_embeds`` at ``image_token_id`` positions so
        the student's forward sees the same vision-grounded embeddings
        as it would at inference. Without this, the student forward
        treats the spliced ``image_token_id`` repetitions as literal
        text tokens — a fatal domain shift from training to eval.
        Symptom (observed 2026-05-28 Stage D-2): GQA 12.3% → 0.67%
        after 50 OPD steps; outputs degenerate to ``"the the the..."``.

        Bypasses ``PolicyTrainer._load_vision_components``'s DCP-load
        path (which expects ``mm_projector.projector.*`` keys but the
        SFT DCP uses ``mm_state.projector.*`` under the 2026-05-14
        ``_MMStateWrapper``). Loads projector weights directly from
        ``{hf_model_path}/model.safetensors``.

        Args:
            hf_model_path: SFT HF VLM dir (e.g.
                ``phase11_rlhf_grpo_infra/hf/stage2_447m_step5200``).
                Must contain ``model.safetensors`` with
                ``mm_projector.projector.{fc1,fc2}.{weight,bias}``.
            vision_tower_id: HF id for SigLIP. Default matches the
                stage-1 LLaVA SFT recipe.

        Returns:
            Diagnostic dict (vision_hidden, llm_hidden, projector_keys_loaded).
        """
        from transformers import SiglipVisionModel
        from safetensors.torch import load_file

        # (1) Vision tower — fresh from HF cache, frozen.
        logger.info(f"OPD vision: loading SigLIP from {vision_tower_id}")
        vt = SiglipVisionModel.from_pretrained(vision_tower_id)
        for p in vt.parameters():
            p.requires_grad = False
        vt.eval()
        vt = vt.to(self.device)
        self._vision_tower = vt
        vision_hidden = vt.config.hidden_size

        # (2) Probe LM hidden size (for projector geometry).
        try:
            llm_hidden = self.model.embed_tokens.weight.shape[1]
        except AttributeError:
            llm_hidden = self.model.tok_embeddings.weight.shape[1]

        # (3) Build 2-layer MLP projector (matches phase5 Projector geometry).
        class _Projector(torch.nn.Module):
            def __init__(self, vd, ld):
                super().__init__()
                self.fc1 = torch.nn.Linear(vd, ld, bias=True)
                self.fc2 = torch.nn.Linear(ld, ld, bias=True)
            def forward(self, x):
                import torch.nn.functional as F
                return self.fc2(F.gelu(self.fc1(x)))
        proj = _Projector(vision_hidden, llm_hidden).to(self.device, torch.bfloat16)

        # (4) Load projector weights from SFT HF safetensors. Keys live
        #     under ``mm_projector.projector.{fc1,fc2}.{weight,bias}``.
        import os as _os
        st_path = _os.path.join(hf_model_path, "model.safetensors")
        if not _os.path.isfile(st_path):
            raise FileNotFoundError(
                f"No model.safetensors at {st_path}; needed for projector weights."
            )
        donor = load_file(st_path)
        proj_sd = {}
        for k, v in donor.items():
            if k.startswith("mm_projector.projector."):
                inner_k = k[len("mm_projector.projector."):]
                proj_sd[inner_k] = v
        proj.load_state_dict(proj_sd, strict=True)
        for p in proj.parameters():
            p.requires_grad = False
        proj.eval()
        self._projector = proj

        logger.info(
            f"OPD vision: vision_tower {vision_hidden}d -> projector -> "
            f"{llm_hidden}d ({len(proj_sd)} projector keys loaded)"
        )
        return {
            "vision_hidden": vision_hidden,
            "llm_hidden": llm_hidden,
            "projector_keys_loaded": len(proj_sd),
        }
