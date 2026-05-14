"""Multimodal full-parameter fine-tune of AttnRes-Kimi (Phase 5).

Subclasses torchtitan's Trainer and threads vision through the standard
PP/FSDP forward-backward path:

* ``post_dataloading_process`` is overridden to: (a) pop ``pixel_values``
  from the dataloader's input_dict, (b) run the frozen vision tower
  under ``no_grad``, (c) run the trainable projector with grad, (d)
  inject ``vision_embeds`` (autograd-live) back into the input_dict so
  the parent's PP/FSDP forward path picks it up via ``extra_inputs``.
* The standard ``forward_backward_step`` then handles both PP and FSDP
  uniformly: stage 0's wrapped ``KimiLinearAttnResModel.forward``
  scatters vision_embeds at IMAGE_TOKEN_ID positions inside its forward
  (the existing path used for the FSDP run since Phase 4e). Middle PP
  stages don't have ``embed_tokens`` and silently ignore the vision
  kwarg. Last stage emits logits, default cross-entropy loss with
  ``ignore_index=-100`` matches the dataset's image/BOS-masked labels.

Why this is the standard layout (vs. a custom ``forward_backward_step``):

* PP scheduler default-chunks every Tensor kwarg along dim 0, so
  ``vision_embeds`` (shape (B, 196, D)) is split per-microbatch
  automatically — same as ``input_ids``. This is the same fixed-shape
  pad pattern Megatron-LM's ``pretrain_vlm.py`` uses for VLM PP.
* The autograd graph from CE loss back through stage 0's scatter into
  ``vision_embeds`` reaches the trainer's projector through PP's
  built-in SEND_B (per-microbatch grads sum into the projector tensor's
  grad).
* No bespoke loss path means the cache adapter's loss-alignment
  comparison (Arm 2's headline metric) reduces to the standard
  PP-vs-FSDP comparison from Phase 3/4.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

WORKSPACE = Path(__file__).resolve().parent.parent
TORCHTITAN_PATH = WORKSPACE / "torchtitan"
for p in (str(WORKSPACE), str(TORCHTITAN_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torchtitan.trainer import Trainer  # noqa: E402
from torchtitan.tools.logging import init_logger, logger  # noqa: E402

# Apply the PP+V≥2+LBS≥2 backward-graph hotfix before any pipeline
# schedule is constructed. Idempotent; safe under non-PP runs (still
# patches the method but it's never invoked). See
# additional_found_issues/torchtitan_pp_lbs_backward_INVESTIGATION.md.
import phase6.torchtitan_pp_backward_hotfix  # noqa: E402,F401
import phase6.torchtitan_pp_retain_graph_diag  # noqa: E402,F401  diagnostic, env-gated
from torchtitan.components.dataloader import ParallelAwareDataloader  # noqa: E402
from torchtitan.distributed import utils as dist_utils  # noqa: E402

from phase5.multimodal_dataset import (  # noqa: E402
    GLOBAL_SEQ_LEN_DEFAULT,
    IGNORE_INDEX,
    IMAGE_TOKEN_ID,
    LlavaPretrainDataset,
    collate_with_pad,
)
from phase5.multimodal_model import Projector  # noqa: E402


# ----------------------------------------------------------------------
# CLI args (consumed before torchtitan's tyro)
# ----------------------------------------------------------------------


def _parse_mm_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mm.json", dest="mm_json", required=True,
                   help="LLaVA-Pretrain caption json path")
    p.add_argument("--mm.images", dest="mm_images", required=True,
                   help="LLaVA-Pretrain images dir")
    p.add_argument("--mm.vision-model", dest="mm_vision_model",
                   default="google/siglip-base-patch16-224")
    p.add_argument("--mm.tokenizer", dest="mm_tokenizer",
                   default="NousResearch/Meta-Llama-3.1-8B",
                   help="Tokenizer source (must match the LM ckpt's pretrain tokenizer)")
    p.add_argument("--mm.cache-dir", dest="mm_cache_dir",
                   default="/root/hf_cache")
    p.add_argument("--mm.proj-lr-mult", dest="mm_proj_lr_mult",
                   type=float, default=1.0,
                   help="LR multiplier for projector params relative to LM LR")
    p.add_argument("--mm.global-seq-len", dest="mm_global_seq_len",
                   type=int, default=GLOBAL_SEQ_LEN_DEFAULT,
                   help="Fixed sequence length for collate (PP P2P shape "
                        "stability). Default 258 = 196 vision + 1 bos + 60 "
                        "caption + 1 eos.")
    p.add_argument("--mm.layout", dest="mm_layout", default="prefix",
                   choices=("prefix", "interior", "random", "sft"),
                   help="Image-token layout policy in input_ids. "
                        "'prefix' (default): original LLaVA layout "
                        "[<img>×196] [BOS] [caption]. "
                        "'interior': image block in middle of caption. "
                        "'random': per-record uniform pick of {prefix, interior}.")
    p.add_argument("--mm.val-samples", dest="mm_val_samples", type=int,
                   default=512,
                   help="Size of the deterministic held-out validation tail "
                        "carved from the end of the caption JSON. Training "
                        "uses records[:-val_samples]; val uses the last "
                        "val_samples records. 0 disables validation.")
    p.add_argument("--mm.val-freq", dest="mm_val_freq", type=int, default=50,
                   help="Run a forward-only validation pass every N training "
                        "steps and log 'val_loss'. Default 50 (frequent so a "
                        "broken run shows up early in smoke runs).")
    p.add_argument("--mm.val-batches", dest="mm_val_batches", type=int,
                   default=24,
                   help="Number of val batches consumed per validation pass "
                        "(forward-only). Caps wall-clock cost of each pass.")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


# ----------------------------------------------------------------------
# Trainer subclass
# ----------------------------------------------------------------------


class MultimodalTrainer(Trainer):
    def __init__(self, config, *,
                 json_path: str, images_dir: str, vision_model: str,
                 tokenizer_path: str, cache_dir: str,
                 proj_lr_mult: float = 1.0,
                 global_seq_len: int = GLOBAL_SEQ_LEN_DEFAULT,
                 layout: str = "prefix",
                 val_samples: int = 512,
                 val_freq: int = 50,
                 val_batches: int = 24):
        super().__init__(config)
        self._mm_layout = layout
        self._val_samples = val_samples
        self._val_freq = val_freq
        self._val_batches = val_batches

        self._global_seq_len = global_seq_len

        # ----- DP rank info -----
        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            dp_world_size = batch_mesh.size()
            dp_rank = batch_mesh.get_local_rank()
        else:
            dp_world_size, dp_rank = 1, 0

        # Detect first-stage rank for PP. Vision tower + projector live on
        # the rank that holds the LM's stage 0 (it's the only place that
        # actually consumes pixel_values + scatters vision into the embed
        # stream). On other PP ranks they're not built — we'd never call
        # them anyway, and their parameters would sit idle hogging memory.
        self._is_vision_rank = (
            (not self.parallel_dims.pp_enabled) or self.pp_has_first_stage
        )
        # For mid/last PP ranks we still want a tokenizer for the dataset
        # (text portion). Vision tower / projector are skipped.

        # ----- Vision tower (frozen, replicated, vision-rank only) -----
        if self._is_vision_rank:
            logger.info(f"mm: loading vision_tower {vision_model}")
            vision = AutoModel.from_pretrained(
                vision_model, dtype=torch.bfloat16,
                cache_dir=cache_dir, low_cpu_mem_usage=True,
            )
            # SigLIP has both vision_model and text_model; we only want vision.
            if hasattr(vision, "vision_model"):
                self.vision_tower = vision.vision_model.to(self.device).eval()
            else:
                self.vision_tower = vision.to(self.device).eval()
            for p in self.vision_tower.parameters():
                p.requires_grad_(False)
            vision_dim = getattr(
                vision.config, "vision_config", vision.config
            ).hidden_size
            logger.info(f"mm: vision_tower hidden_size={vision_dim}, frozen")

            self.image_processor = AutoProcessor.from_pretrained(
                vision_model, cache_dir=cache_dir,
            )
        else:
            self.vision_tower = None
            self.image_processor = None
            vision_dim = None

        self.mm_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, cache_dir=cache_dir,
        )

        # Resolve image-sentinel id from the per-tokenizer registry rather
        # than the legacy hardcoded IMAGE_TOKEN_ID. The registry warns at
        # startup if a non-reserved sentinel collides too often with real
        # caption text. Sample 1000 captions from the dataset for the
        # check (skipped on non-vision ranks where the dataset isn't
        # consulted yet at this point).
        from phase5.sentinel_registry import resolve_sentinel  # noqa: E402
        try:
            self._image_sentinel_id = resolve_sentinel(
                self.mm_tokenizer,
                role="image",
                sample_captions=None,  # full check happens during dataset construction
                strict=False,
            )
            logger.info(
                f"mm: image sentinel resolved via registry → id={self._image_sentinel_id} "
                f"(legacy hardcoded value was 32_000)"
            )
        except Exception as e:
            logger.warning(
                f"mm: sentinel registry resolution failed ({e}); "
                f"falling back to legacy IMAGE_TOKEN_ID=32_000"
            )
            self._image_sentinel_id = IMAGE_TOKEN_ID

        # ----- LM dim (Kimi 436M = 1168) -----
        # On non-first PP stages the LM submodule may not have embed_tokens,
        # so we read hidden_size from the model's config instead.
        lm = self.model_parts[0]
        lm_dim = getattr(getattr(lm, "config", None), "hidden_size", None)
        if lm_dim is None:
            # Fallback: try to read from embed_tokens (first stage / non-PP).
            if hasattr(lm, "embed_tokens") and lm.embed_tokens is not None:
                lm_dim = lm.embed_tokens.weight.shape[1]
        if lm_dim is None:
            raise RuntimeError(
                "Cannot infer lm_dim from model_parts[0].config or .embed_tokens"
            )
        self._lm_dim = lm_dim
        logger.info(f"mm: lm_dim={lm_dim}")

        # ----- Projector (trainable, vision-rank only) -----
        if self._is_vision_rank:
            self.projector = Projector(vision_dim=vision_dim, lm_dim=lm_dim).to(
                device=self.device, dtype=torch.bfloat16,
            )
            n_proj_params = sum(p.numel() for p in self.projector.parameters())
            logger.info(f"mm: projector built, params={n_proj_params:,}")

            # FSDP2 wrap on the batch/dp mesh so projector grads are
            # reduce-scattered across DP ranks. Without this wrap, each
            # FSDP rank's projector trains on its own dp shard's samples
            # only — projector copies silently diverge across ranks, and
            # rank 0's projector sees ~1/dp_world_size of the actual
            # batch per step. This was the FSDP=4 vs PP=4 alignment
            # divergence root cause: PP rank 0 has the only projector
            # and sees the full per-step batch (correct); FSDP=4 has a
            # projector per rank, each seeing 1/4 of the batch (broken)
            # → FSDP loss curve trains slower than PP for a multimodal
            # property other than parallelism strategy.
            #
            # Under PP-only (no dp axis), get_optional_mesh("batch")
            # returns None and we leave the projector unwrapped — the
            # single PP-rank-0 projector already sees the full batch.
            batch_mesh = self.parallel_dims.get_optional_mesh("batch")
            if batch_mesh is not None and batch_mesh.size() > 1:
                from torch.distributed._composable.fsdp import fully_shard
                fully_shard(self.projector, mesh=batch_mesh)
                logger.info(
                    f"mm: projector wrapped with FSDP2 over batch mesh "
                    f"size={batch_mesh.size()} (grad sync across DP ranks)"
                )
            else:
                logger.info(
                    "mm: projector unwrapped (no DP axis; single-rank projector)"
                )

            # Separate AdamW for the projector — torchtitan's LambdaLR was
            # built for the LM's original param groups only and asserts
            # strict zip(groups, lr_values), so we can't add_param_group.
            #
            # IMPORTANT: we deliberately do NOT append this to
            # ``OptimizersContainer.optimizers``. The container's
            # ``state_dict()`` does ``map(get_optimizer_state_dict,
            # self.model_parts, self.optimizers)`` — ``map`` truncates to
            # the shorter iterable, so with model_parts=[lm] and
            # optimizers=[lm_opt, proj_opt] the projector optimizer is
            # silently DROPPED from every checkpoint (and if model_parts
            # ever had ≥2 entries — e.g. PP — the projector optimizer
            # would instead be paired with the LM model and crash inside
            # ``get_optimizer_state_dict`` with ``KeyError: 0`` because
            # the projector's params are absent from the LM's FQN map).
            # The projector is stepped explicitly in ``train_step`` and
            # checkpointed via the dedicated ``_MMStateWrapper`` below.
            proj_lr = config.optimizer.lr * proj_lr_mult
            proj_optim = torch.optim.AdamW(
                list(self.projector.parameters()),
                lr=proj_lr,
                betas=(0.9, 0.95), weight_decay=0.01,
            )
            self._proj_optim = proj_optim
            logger.info(
                f"mm: built standalone projector AdamW (lr={proj_lr}, fixed); "
                f"stepped explicitly in train_step, not via OptimizersContainer"
            )

            # Register projector + its optimizer + the LM optimizer with
            # the checkpointer via a single robust wrapper.
            #
            # Why we also override the LM optimizer's checkpoint entry:
            # ``OptimizersContainer.state_dict()`` calls upstream
            # ``get_optimizer_state_dict(lm_model, lm_optim)``, which
            # builds an FQN→param-id map by *object identity* between
            # ``lm_optim.param_groups`` and ``lm_model.named_parameters()``.
            # With ``tie_word_embeddings=True`` the embed/lm_head weight is
            # a shared tensor; the bundled FSDP2 wrap + a subsequent
            # ``initial_load_model_only`` checkpoint load can leave the
            # optimizer holding a parameter object that is no longer the
            # one yielded by ``named_parameters()`` → ``KeyError: 0`` on
            # the tied embedding's param-id, crashing every step-N save.
            # ``_robust_optim_state_dict`` rebuilds that mapping at save
            # time and falls back to positional FQN matching, which is
            # correct for the full-parameter SFT param ordering and
            # immune to identity drift.
            from torch.distributed.checkpoint.stateful import Stateful  # noqa: E402

            lm_model = self.model_parts[0]

            def _robust_optim_state_dict(model, optim):
                """FQN-keyed optimizer state_dict immune to param-identity
                drift between ``optim.param_groups`` and
                ``model.named_parameters()`` (tied weights / FSDP bundle /
                post-load reassignment)."""
                from itertools import chain

                osd = optim.state_dict()  # {"state": {pid: ...}, "param_groups": [...]}
                opt_params = list(
                    chain.from_iterable(
                        g["params"] for g in optim.param_groups
                    )
                )
                model_named = list(model.named_parameters())
                id_to_fqn = {id(p): n for n, p in model_named}
                # pid → FQN, by identity first, positional fallback second.
                pid_to_fqn = {}
                for pid, p in enumerate(opt_params):
                    fqn = id_to_fqn.get(id(p))
                    if fqn is None and pid < len(model_named):
                        fqn = model_named[pid][0]
                    if fqn is None:
                        raise RuntimeError(
                            f"mm: cannot resolve FQN for optimizer param "
                            f"#{pid} (model has {len(model_named)} named "
                            f"params, optimizer has {len(opt_params)})"
                        )
                    pid_to_fqn[pid] = fqn
                state_by_fqn = {
                    pid_to_fqn[pid]: st for pid, st in osd["state"].items()
                }
                pgs = []
                for g in osd["param_groups"]:
                    g = dict(g)
                    g["params"] = [pid_to_fqn[pid] for pid in g["params"]]
                    pgs.append(g)
                return {"state": state_by_fqn, "param_groups": pgs}

            def _robust_load_optim_state_dict(model, optim, sd):
                """Inverse of ``_robust_optim_state_dict``: map FQN-keyed
                state back onto the optimizer's current param ids."""
                from itertools import chain

                opt_params = list(
                    chain.from_iterable(
                        g["params"] for g in optim.param_groups
                    )
                )
                model_named = list(model.named_parameters())
                id_to_fqn = {id(p): n for n, p in model_named}
                fqn_to_pid = {}
                for pid, p in enumerate(opt_params):
                    fqn = id_to_fqn.get(id(p))
                    if fqn is None and pid < len(model_named):
                        fqn = model_named[pid][0]
                    if fqn is not None:
                        fqn_to_pid[fqn] = pid
                state_by_pid = {
                    fqn_to_pid[fqn]: st
                    for fqn, st in sd["state"].items()
                    if fqn in fqn_to_pid
                }
                pgs = []
                for g in sd["param_groups"]:
                    g = dict(g)
                    g["params"] = [
                        fqn_to_pid[fqn]
                        for fqn in g["params"]
                        if fqn in fqn_to_pid
                    ]
                    pgs.append(g)
                optim.load_state_dict({"state": state_by_pid, "param_groups": pgs})

            class _MMStateWrapper(Stateful):
                """Single checkpointer entry that robustly saves/loads the
                projector weights, the projector optimizer, AND the LM
                optimizer (replacing the brittle ``OptimizersContainer``
                ``OPTIMIZER`` entry — see class comment above)."""

                def __init__(self_w, projector, proj_optim, lm_model, lm_optim):
                    self_w.projector = projector
                    self_w.proj_optim = proj_optim
                    self_w.lm_model = lm_model
                    self_w.lm_optim = lm_optim

                def state_dict(self_w):
                    from torch.distributed.checkpoint.state_dict import (
                        get_model_state_dict,
                    )
                    return {
                        "projector": get_model_state_dict(self_w.projector),
                        "proj_optim": _robust_optim_state_dict(
                            self_w.projector, self_w.proj_optim,
                        ),
                        "lm_optim": _robust_optim_state_dict(
                            self_w.lm_model, self_w.lm_optim,
                        ),
                    }

                def load_state_dict(self_w, sd):
                    from torch.distributed.checkpoint.state_dict import (
                        set_model_state_dict,
                    )
                    if "projector" in sd:
                        set_model_state_dict(
                            self_w.projector, model_state_dict=sd["projector"],
                        )
                    if "proj_optim" in sd:
                        _robust_load_optim_state_dict(
                            self_w.projector, self_w.proj_optim, sd["proj_optim"],
                        )
                    if "lm_optim" in sd:
                        _robust_load_optim_state_dict(
                            self_w.lm_model, self_w.lm_optim, sd["lm_optim"],
                        )

            if hasattr(self, "checkpointer") and self.checkpointer is not None:
                # The container holds exactly one inner optimizer (the LM
                # AdamW) — we route its checkpoint through the robust
                # wrapper and drop the container's brittle OPTIMIZER entry.
                lm_optim = self.optimizers.optimizers[0]
                self.checkpointer.states["mm_state"] = _MMStateWrapper(
                    self.projector, proj_optim, lm_model, lm_optim,
                )
                # Remove the OptimizersContainer's OPTIMIZER entry so DCP
                # never calls its identity-fragile state_dict(). Safe: the
                # LM optimizer state is now fully covered by "mm_state".
                self.checkpointer.states.pop("optimizer", None)
                logger.info(
                    "mm: registered _MMStateWrapper (projector + proj_optim "
                    "+ lm_optim) with checkpointer; dropped brittle "
                    "OptimizersContainer OPTIMIZER entry"
                )
        else:
            self.projector = None
            self._proj_optim = None

        # ----- Replace dataloader -----
        # Tokenizer is enough on every rank; the image processor is only
        # needed on the vision rank, but the dataset needs an image_processor
        # to preprocess pixel_values for that rank. On non-vision ranks,
        # we still build a dataset with the same processor (lazy-loaded on
        # vision rank) to keep tokenization deterministic across DP shards.
        if self.image_processor is None:
            # Build a temporary processor purely for non-vision ranks so
            # the dataset can still emit pixel_values placeholders. They'll
            # be ignored downstream (mid/last PP stages don't consume
            # pixel_values), but the collator needs a tensor in the dict.
            self.image_processor = AutoProcessor.from_pretrained(
                vision_model, cache_dir=cache_dir,
            )

        pad_id = self.mm_tokenizer.pad_token_id or 0
        # Stash dataset-construction params so _build_mm_dataset can build
        # both the train dataloader and (later) the held-out val dataloader
        # with identical tokenization / DP-sharding behavior.
        self._mm_ds_kwargs = dict(
            json_path=json_path,
            images_dir=images_dir,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        self._mm_pad_id = pad_id
        self._mm_local_bs = config.training.local_batch_size

        # ----- train dataloader -----
        # The "sft" layout uses phase9's LlavaInstructSFTDataset which has no
        # held-out split support → validation is disabled for that layout.
        if self._mm_layout == "sft" and self._val_samples > 0:
            logger.warning(
                "mm: val: sft layout does not support a held-out split; "
                "disabling validation (set --mm.val-samples 0 to silence)."
            )
            self._val_samples = 0

        train_ds = self._build_mm_dataset(split="train", infinite=True)
        self.dataloader = ParallelAwareDataloader(
            train_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=self._mm_local_bs,
            collate_fn=lambda b: collate_with_pad(
                b, pad_id=pad_id, global_seq_len=self._global_seq_len,
            ),
            num_workers=0,
        )
        logger.info(
            f"mm: replaced dataloader (train split, "
            f"N={len(train_ds.records):,}, dp_rank={dp_rank}/{dp_world_size}, "
            f"local_bs={self._mm_local_bs}, seq_len={self._global_seq_len})"
        )

        # ----- held-out val dataloader (forward-only, single-pass) -----
        self._val_dataloader = None
        if self._val_samples > 0 and self._val_freq > 0:
            val_ds = self._build_mm_dataset(split="val", infinite=True)
            # infinite=True on the dataset so a val pass capped at
            # _val_batches never raises StopIteration mid-pass even when the
            # per-rank val shard is smaller than _val_batches * local_bs.
            self._val_dataloader = ParallelAwareDataloader(
                val_ds,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                batch_size=self._mm_local_bs,
                collate_fn=lambda b: collate_with_pad(
                    b, pad_id=pad_id, global_seq_len=self._global_seq_len,
                ),
                num_workers=0,
            )
            logger.info(
                f"mm: val enabled — held-out tail N={len(val_ds.records):,}, "
                f"val_freq={self._val_freq}, val_batches={self._val_batches} "
                f"(disjoint from train split)"
            )
        else:
            logger.info("mm: val disabled (val_samples<=0 or val_freq<=0)")

    def _build_mm_dataset(self, *, split: str, infinite: bool):
        """Construct the layout-appropriate MM dataset for a given split.

        ``split`` ('train'|'val') selects the disjoint record range; the
        held-out val tail is deterministic and never seen by training.
        ``sft`` layout has no split support and only ever uses 'train'.
        """
        k = self._mm_ds_kwargs
        if self._mm_layout == "sft":
            from phase9.multimodal_sft_dataset import LlavaInstructSFTDataset
            ds = LlavaInstructSFTDataset(
                tokenizer=self.mm_tokenizer,
                image_processor=self.image_processor,
                **k,
            )
            logger.info(
                "mm: dataset = LlavaInstructSFTDataset (sft layout, conversation format)"
            )
        elif self._mm_layout == "prefix":
            ds = LlavaPretrainDataset(
                tokenizer=self.mm_tokenizer,
                image_processor=self.image_processor,
                split=split,
                val_samples=self._val_samples,
                infinite=infinite,
                **k,
            )
            logger.info(f"mm: dataset = LlavaPretrainDataset (prefix layout, split={split})")
        else:
            from phase5.multimodal_dataset_interleave import (
                InterleavedLlavaPretrainDataset,
            )
            ds = InterleavedLlavaPretrainDataset(
                tokenizer=self.mm_tokenizer,
                image_processor=self.image_processor,
                layout=self._mm_layout,
                split=split,
                val_samples=self._val_samples,
                infinite=infinite,
                **k,
            )
            logger.info(
                f"mm: dataset = InterleavedLlavaPretrainDataset "
                f"(layout={self._mm_layout!r}, split={split})"
            )
        return ds

    # ------------------------------------------------------------------
    # Multimodal injection point: post_dataloading_process.
    # ------------------------------------------------------------------
    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """Pop pixel_values, compute vision_embeds, inject into input_dict.

        The trainer's standard PP/FSDP forward-backward path then treats
        ``vision_embeds`` as a regular ``extra_input`` (kwarg passed to the
        first stage). The PP scheduler default-chunks tensor kwargs along
        dim 0, so each microbatch's ``vision_embeds`` slice lines up with
        its ``input_ids`` slice.
        """
        if "pixel_values" in input_dict:
            pixel_values = input_dict.pop("pixel_values")
        else:
            pixel_values = None

        # Only the vision rank actually computes the projection. Mid/last
        # PP ranks won't see ``pixel_values`` from this DP shard's dataloader
        # because torchtitan's batch_generator is constructed on every rank
        # but its outputs are consumed only where the model needs them.
        # Even if a mid-rank dataloader yields pixel_values, we just drop it.
        if pixel_values is not None and self._is_vision_rank and self.vision_tower is not None:
            pixel_values = pixel_values.to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True,
            )
            with torch.no_grad():
                vision_out = self.vision_tower(pixel_values=pixel_values)
                vision_features = vision_out.last_hidden_state  # (B, N_vis, V_dim)
            # Compute the projector output, then sever its autograd graph
            # from PP. Why: each PP microbatch's vision_embeds slice routes
            # grad back to the SAME projector grad_fn (the projector runs
            # ONCE per step, on the full pre-microbatch batch). With V>=2
            # + LBS>=3 + Interleaved1F1B, two microbatches' stage_backward
            # calls hit that shared grad_fn and the second one crashes with
            # "Trying to backward through the graph a second time". Detach
            # makes ``vision_embeds_leaf`` a fresh leaf (no upstream graph)
            # so PP's per-microbatch backward only walks back to the leaf.
            # The leaf accumulates ``.grad`` across microbatches via
            # AccumulateGrad; ``forward_backward_step`` below replays a
            # single backward through the projector with the summed grad
            # so the projector still trains correctly.
            vision_embeds_orig = self.projector(vision_features)
            vision_embeds_leaf = (
                vision_embeds_orig.detach().requires_grad_(True)
            )
            input_dict["vision_embeds"] = vision_embeds_leaf
            self._mm_projector_stash = (vision_embeds_orig, vision_embeds_leaf)

        # Hand off to the parent for the standard inputs/extra_inputs/extra_kwargs split.
        return super().post_dataloading_process(input_dict, labels)

    def forward_backward_step(
        self, *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Wraps the trainer's PP/FSDP forward+backward to drive the
        deferred projector backward.

        ``post_dataloading_process`` detaches the projector output before
        handing it to PP (see the comment there for why). After PP /
        FSDP backward returns, ``vision_embeds_leaf.grad`` holds the
        sum of every microbatch's slice grad. We replay a single
        backward through the original projector graph with that summed
        grad so the projector parameters receive the correct accumulated
        gradient and FSDP's reduce-scatter on them sees the full step.
        """
        self._mm_projector_stash = None
        loss = super().forward_backward_step(
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
        )
        stash = getattr(self, "_mm_projector_stash", None)
        if stash is not None:
            vision_embeds_orig, vision_embeds_leaf = stash
            grad = vision_embeds_leaf.grad
            if grad is not None:
                torch.autograd.backward(vision_embeds_orig, grad)
            self._mm_projector_stash = None
        return loss

    # ------------------------------------------------------------------
    # Held-out validation: periodic forward-only val loss.
    # ------------------------------------------------------------------
    def train_step(self, data_iterator):
        """Run the standard train step, then maybe a forward-only val pass.

        ``self.step`` was already incremented by ``Trainer.train()`` before
        this is called, so ``self.step`` here is the just-completed step.
        Validation runs *after* optimizers.step()/zero_grad() inside
        ``train_step``, so grad state is already clean before the val pass.

        The projector optimizer is NOT in the OptimizersContainer (see the
        __init__ comment on why), so it is stepped here explicitly. The
        deferred projector backward in ``forward_backward_step`` accumulates
        grads into the projector params across all microbatches of this
        step; we step once and zero once, mirroring the container's
        per-step step()/zero_grad() cadence for the LM optimizer.
        """
        super().train_step(data_iterator)
        proj_optim = getattr(self, "_proj_optim", None)
        if proj_optim is not None:
            proj_optim.step()
            proj_optim.zero_grad(set_to_none=True)
        if (
            self._val_dataloader is not None
            and self._val_freq > 0
            and (self.step == 1 or self.step % self._val_freq == 0)
        ):
            self._run_validation(self.step)

    @torch.no_grad()
    def _run_validation(self, step: int) -> None:
        """Forward-only mean cross-entropy on the held-out val split.

        Pure 1D FSDP only (pp=1/tp=1). All DP ranks run the same number of
        val batches in lockstep so the dist_sum collective at the end never
        hangs. Optimizer / RNG / dataloader-position / grad state are all
        left untouched: we never call backward, never step the optimizer,
        never touch the training dataloader, and reseed nothing. The model
        is flipped to eval() for the pass and restored to train() after.
        """
        if self.parallel_dims.pp_enabled:
            # Phase 5 is pure-1D-FSDP by design; PP val would need the
            # pp_schedule.eval() path. Skip rather than risk a hang.
            logger.warning("mm: val: PP enabled — skipping val pass (pure-FSDP only)")
            return

        parallel_dims = self.parallel_dims
        model = self.model_parts[0]
        was_training = model.training
        vt_was_training = (
            self.vision_tower.training if self.vision_tower is not None else None
        )
        proj_was_training = (
            self.projector.training if self.projector is not None else None
        )
        model.eval()
        if self.vision_tower is not None:
            self.vision_tower.eval()
        if self.projector is not None:
            self.projector.eval()

        local_loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        local_valid_tokens = torch.zeros((), dtype=torch.int64, device=self.device)
        n_batches = 0

        # Fresh iterator over the val dataloader each pass — the training
        # dataloader's iterator (data_iterator in train()) is a separate
        # object and is never advanced here.
        val_iter = iter(self._val_dataloader)
        try:
            for _ in range(self._val_batches):
                try:
                    input_dict, labels = next(val_iter)
                except StopIteration:
                    break
                for k, v in input_dict.items():
                    if isinstance(v, torch.Tensor):
                        input_dict[k] = v.to(self.device)
                labels = labels.to(self.device)

                self._mm_projector_stash = None
                # Reuse the MM post_dataloading_process: it injects
                # vision_embeds exactly as in training. Under no_grad the
                # detached leaf carries no graph; we clear the stash after.
                inputs, labels, extra_inputs, extra_kwargs = (
                    self.post_dataloading_process(input_dict, labels)
                )
                self._mm_projector_stash = None

                with self.train_context():
                    pred = model(inputs, **extra_inputs, **extra_kwargs)
                    loss_sum = self.loss_fn(pred, labels)
                del pred

                local_loss_sum += loss_sum.detach().float()
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                n_batches += 1
        finally:
            # Always restore the original train/eval modes, even on error.
            if was_training:
                model.train()
            if vt_was_training:
                self.vision_tower.train()
            if proj_was_training:
                self.projector.train()
            self._mm_projector_stash = None

        # Reduce loss-sum and token-count across DP ranks, then divide so
        # the reported number is a true global mean CE (matches training's
        # loss accounting). dist_sum runs on every rank → no hang.
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_loss_sum = dist_utils.dist_sum(local_loss_sum, batch_mesh)
            global_valid_tokens = dist_utils.dist_sum(
                local_valid_tokens, batch_mesh
            )
        else:
            global_loss_sum = float(local_loss_sum.item())
            global_valid_tokens = float(local_valid_tokens.item())

        if global_valid_tokens > 0:
            val_loss = float(global_loss_sum) / float(global_valid_tokens)
        else:
            val_loss = float("nan")

        # Log under a grep-able key. metrics_processor.log_validation also
        # writes 'validation_metrics/loss' to TB.
        logger.info(f"mm: val_loss={val_loss:.4f} step={step} (n_batches={n_batches})")
        if hasattr(self, "metrics_processor") and self.metrics_processor is not None:
            try:
                self.metrics_processor.log_validation(loss=val_loss, step=step)
            except Exception as e:  # pragma: no cover - logging best-effort
                logger.warning(f"mm: val: metrics_processor.log_validation failed: {e}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main():
    init_logger()
    mm_args = _parse_mm_args()

    from torchtitan.config import ConfigManager
    cm = ConfigManager()
    config = cm.parse_args(sys.argv[1:])

    trainer = MultimodalTrainer(
        config,
        json_path=mm_args.mm_json,
        images_dir=mm_args.mm_images,
        vision_model=mm_args.mm_vision_model,
        tokenizer_path=mm_args.mm_tokenizer,
        cache_dir=mm_args.mm_cache_dir,
        proj_lr_mult=mm_args.mm_proj_lr_mult,
        global_seq_len=mm_args.mm_global_seq_len,
        layout=mm_args.mm_layout,
        val_samples=mm_args.mm_val_samples,
        val_freq=mm_args.mm_val_freq,
        val_batches=mm_args.mm_val_batches,
    )
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
