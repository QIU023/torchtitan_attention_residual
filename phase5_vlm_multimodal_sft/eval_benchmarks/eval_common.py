"""Shared eval runner for VLM downstream benchmarks.

Reuses ``MultimodalTrainer`` to load the DCP checkpoint (vision tower +
projector + LM, all sharded by 1D FSDP=8) and exposes a generation API
(``EvalRunner.generate``) that mirrors the training-time
``post_dataloading_process`` + LM forward but runs greedy decode for
``max_new_tokens`` instead of computing CE loss.

Why subclass ``MultimodalTrainer`` instead of writing a fresh loader:

* The trainer's ``__init__`` is the only code that correctly:
    - parallel-dims setup (1D FSDP=8 mesh)
    - vision_tower load + projector build + FSDP wrap
    - DCP checkpoint resolve (initial_load_path + initial_load_model_only +
      the robust ``_MMStateWrapper`` that restores LM + projector + LM-optim)
    - tokenizer + image_processor wiring identical to training
  Rewriting any of this risks subtle drift (different sentinel id, different
  projector dtype, wrong FSDP mesh) → silent wrong predictions.

* We skip ``trainer.train()`` entirely and instead call our own
  ``EvalRunner.run()`` after the checkpoint has been loaded.

Per-rank work distribution:
  rank r processes ``records[r::world_size]``. Each rank writes its
  predictions to ``<output_dir>/preds_rank{r}.jsonl``. After all ranks
  finish (barrier), rank 0 reads all files, dedup-merges, runs the scorer,
  and writes ``<output_dir>/result.json`` + appends a row to REPORT.md.

Generation:
  greedy (argmax), single batch step at a time. We do NOT use any KV cache —
  KimiLinear's KDA + MLA + AttnRes path does not expose a generation-cache
  interface in torchtitan, and implementing one risks correctness regression
  near the freshly trained ckpt. For max_new_tokens=32 on bs=1 this is ~30
  forward passes per sample; with ~5K samples per benchmark and 8 GPUs
  splitting work, that's still tractable inside the 2-3h budget. See
  per-benchmark time estimates in REPORT.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
from PIL import Image

WORKSPACE = Path(__file__).resolve().parent.parent.parent
TORCHTITAN_PATH = WORKSPACE / "torchtitan"
for p in (str(WORKSPACE), str(TORCHTITAN_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from phase5_vlm_multimodal_sft.multimodal_dataset import (  # noqa: E402
    IGNORE_INDEX, N_VISION_TOKENS,
)
from phase5_vlm_multimodal_sft.train_mm import MultimodalTrainer  # noqa: E402
from torchtitan.tools.logging import init_logger, logger  # noqa: E402


# --------------------------------------------------------------------------
# CKPT-loaded trainer: minimal subclass that swaps out train()
# --------------------------------------------------------------------------
class _LoadedTrainer(MultimodalTrainer):
    """Same init as MultimodalTrainer (loads ckpt via initial_load_path),
    but does not run train(). Caller invokes load() then uses the trainer
    as a stateful evaluator.
    """

    def load_ckpt_only(self):
        """Trigger the checkpointer's initial-load path so the LM +
        projector weights are restored from the DCP folder.
        Mirrors what ``Trainer.train()`` does on its first iteration."""
        # Step counter must be 0 so the checkpointer treats this as an
        # initial-load (not a resume).
        self.step = 0
        loaded = self.checkpointer.load(step=self.step)
        if not loaded:
            raise RuntimeError(
                f"checkpointer.load() returned False — "
                f"DCP folder at {self.checkpointer.initial_load_path!r} "
                f"could not be loaded. Verify the directory contains "
                f"__*_*.distcp files and was produced by torchtitan."
            )
        # Set models to eval mode globally.
        for part in self.model_parts:
            part.eval()
        if self.vision_tower is not None:
            self.vision_tower.eval()
        if self.projector is not None:
            self.projector.eval()
        logger.info(f"eval: ckpt loaded from {self.checkpointer.initial_load_path}")


# --------------------------------------------------------------------------
# Public façade: EvalRunner
# --------------------------------------------------------------------------
class EvalRunner:
    """Build once, score one benchmark at a time, write per-rank JSONL."""

    def __init__(self, trainer: _LoadedTrainer, max_new_tokens: int = 32):
        self.trainer = trainer
        self.max_new_tokens = max_new_tokens
        # Cached state from the trainer for fast access in the hot loop.
        self.device = trainer.device
        self.tokenizer = trainer.mm_tokenizer
        self.image_processor = trainer.image_processor
        self.vision_tower = trainer.vision_tower
        self.projector = trainer.projector
        self.lm = trainer.model_parts[0]
        self.image_sentinel = trainer._image_sentinel_id

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Resolve EOS so we can early-stop generation.
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = 128_001  # Llama-3.1 default
        self.eos_id = eos
        self.bos_id = self.tokenizer.bos_token_id or 128_000
        # Newline id (used to stop on multi-line generations for MCQ).
        try:
            nl = self.tokenizer.encode("\n", add_special_tokens=False)
            self.newline_id = nl[0] if nl else None
        except Exception:
            self.newline_id = None

    # ----- prompt construction matching the SFT training format -----
    def build_input_ids(self, question: str) -> list[int]:
        """[<img>*196][BOS][USER: <q>\n][ASSISTANT:]

        Matches the multi-turn SFT format from
        ``phase9_post_training_ppo_trace.multimodal_sft_dataset``:
            "USER: " + text + "\n"      (for human turn)
            "ASSISTANT: " + text        (for gpt turn, no trailing \n)
        At inference we want the model's first emitted token to be the
        first content token of the assistant turn. In Llama-3.1 BPE the
        gpt-turn answer's leading space is glued onto its first token
        (e.g. ' The', ' yes', ' A'). The training dataset builds the
        prefix as ``self.tokenizer.encode("ASSISTANT: " + text + ...)``
        which produces ``[..., ':' = 25, ' The' = 578, ...]`` — the
        ' ' (id 220) sits as the leading space of the first content
        token, NOT as a separate id. So at inference we must stop the
        prompt at the ':' (id 25) and let the model emit the leading
        ' yes' / ' A' / ' The' itself; appending a literal " " (id 220)
        would push the prompt off-distribution.

        We achieve this by tokenizing the user turn + the literal string
        "ASSISTANT:" (no trailing space), then letting greedy decode
        produce ' yes' / ' A' / ' The' as the first new token.
        """
        user_text = "USER: " + question.replace("<image>", "").strip() + "\n"
        asst_prefix = "ASSISTANT:"  # no trailing space — see docstring
        user_ids = self.tokenizer.encode(user_text, add_special_tokens=False)
        asst_ids = self.tokenizer.encode(asst_prefix, add_special_tokens=False)
        return (
            [self.image_sentinel] * N_VISION_TOKENS
            + [self.bos_id]
            + user_ids
            + asst_ids
        )

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int | None = None,
                 stop_on_newline: bool = False) -> str:
        """Greedy decode for one sample."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # 1) Build prompt tokens.
        prompt_ids = self.build_input_ids(question)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # 2) Vision forward → projector (matches training).
        pixel = self.image_processor(
            images=image.convert("RGB"), return_tensors="pt",
        )["pixel_values"].to(device=self.device, dtype=torch.bfloat16)
        vision_features = self.vision_tower(pixel_values=pixel).last_hidden_state
        vision_embeds = self.projector(vision_features)  # (1, 196, D)
        # image_mask is recomputed from input_ids inside the LM forward
        # using the sentinel id; we pass it explicitly to be safe.
        image_mask = (input_ids == self.image_sentinel)

        generated: list[int] = []
        # 3) Greedy decode: rebuild the full prefix each step. KimiLinear /
        # AttnRes has no kv-cache exposed in torchtitan; for short
        # max_new_tokens (≤64) this is acceptable and matches the model's
        # train-time forward exactly.
        for _ in range(max_new_tokens):
            # IMPORTANT: pad vision_embeds slot count is fixed at
            # N_VISION_TOKENS in the prompt; the appended generated tokens
            # have NO image sentinels so image_mask stays 196-wide on the
            # first axis but expanded over the longer T dim.
            cur_ids = torch.cat([
                input_ids,
                torch.tensor([generated], dtype=torch.long, device=self.device)
                if generated else torch.empty((1, 0), dtype=torch.long, device=self.device),
            ], dim=1)
            cur_mask = (cur_ids == self.image_sentinel)
            logits = self.lm(
                cur_ids,
                vision_embeds=vision_embeds,
                image_mask=cur_mask,
            )
            # Last-token logits: (1, V).
            next_id = int(logits[0, -1].argmax().item())
            if next_id == self.eos_id:
                break
            if stop_on_newline and self.newline_id is not None and next_id == self.newline_id:
                break
            generated.append(next_id)
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    # ----- distributed I/O -----
    def my_indices(self, n_total: int) -> list[int]:
        """Round-robin shard of indices for this rank.

        IMPORTANT: pads all ranks to ``ceil(n_total/world_size)`` length to
        keep FSDP collectives in lock-step. Ranks whose padded slots have
        no real record receive ``-1`` and must skip the actual generation
        (but still call the LM forward with a dummy one-token sample to
        keep the all-gather collectives in step with the busy ranks).

        Without padding, the rank that runs out of records first stops
        calling ``self.lm(...)`` and stalls every other rank inside FSDP's
        all-gather. With padding, every rank issues exactly the same
        number of forward passes; the late-finishing ranks then write
        their preds files in true lockstep.
        """
        ws = self.world_size
        max_per_rank = (n_total + ws - 1) // ws  # ceil
        idx: list[int] = []
        for k in range(max_per_rank):
            global_i = self.rank + k * ws
            idx.append(global_i if global_i < n_total else -1)
        return idx

    def write_preds(self, output_dir: str, preds: list[dict[str, Any]]):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"preds_rank{self.rank}.jsonl"
        with open(p, "w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")
        logger.info(f"eval: rank {self.rank} wrote {len(preds)} preds → {p}")

    def barrier_then_gather_preds(self, output_dir: str,
                                  max_wait_sec: int = 1800) -> list[dict[str, Any]]:
        """File-based fan-in (NO NCCL barrier).

        Rank 0 polls the output dir until every per-rank file exists OR
        the deadline is hit. We deliberately avoid ``dist.barrier()`` here
        because NCCL collectives time out at the default 10-minute mark
        when one rank is much slower (e.g. uneven shard of long
        generations), tearing down the whole process group with SIGABRT.
        With filesystem coordination, rank 0 just waits; non-rank-0
        returns immediately after writing its file.
        """
        out = Path(output_dir)
        if self.rank != 0:
            return []
        deadline = time.time() + max_wait_sec
        missing = set(range(self.world_size))
        while missing and time.time() < deadline:
            for r in list(missing):
                p = out / f"preds_rank{r}.jsonl"
                if p.exists():
                    # also wait for file to be at least 1 line stable
                    missing.discard(r)
            if missing:
                time.sleep(1.0)
        if missing:
            logger.warning(f"eval: timed out waiting for ranks {sorted(missing)}")
        all_records: list[dict[str, Any]] = []
        for r in range(self.world_size):
            p = out / f"preds_rank{r}.jsonl"
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_records.append(json.loads(line))
                    except Exception as e:
                        logger.warning(f"eval: bad jsonl line in {p}: {e}")
        return all_records


# --------------------------------------------------------------------------
# Bootstrapping: build a _LoadedTrainer from the standard launcher CLI.
# --------------------------------------------------------------------------
def build_trainer_from_args() -> _LoadedTrainer:
    """Mirror train_mm.main(): parse mm args + torchtitan ConfigManager,
    build the trainer, then load the ckpt and return it ready to eval.
    """
    init_logger()
    # ----- 1) MM args (consumed before tyro) -----
    from phase5_vlm_multimodal_sft.train_mm import _parse_mm_args  # noqa: E402
    mm_args = _parse_mm_args()

    # ----- 2) torchtitan config -----
    from torchtitan.config import ConfigManager
    cm = ConfigManager()
    config = cm.parse_args(sys.argv[1:])

    # ----- 3) instantiate trainer (no train()) -----
    trainer = _LoadedTrainer(
        config,
        json_path=mm_args.mm_json,
        images_dir=mm_args.mm_images,
        vision_model=mm_args.mm_vision_model,
        tokenizer_path=mm_args.mm_tokenizer,
        cache_dir=mm_args.mm_cache_dir,
        proj_lr_mult=mm_args.mm_proj_lr_mult,
        global_seq_len=mm_args.mm_global_seq_len,
        layout=mm_args.mm_layout,
        val_samples=0,
        val_freq=0,
        val_batches=0,
        freeze_lm=False,
        text_len=mm_args.mm_text_len,
        shuffle_seed=0,
        val_stratified_per_source=0,
    )
    trainer.load_ckpt_only()
    return trainer


# --------------------------------------------------------------------------
# Convenience: end-to-end driver for a single benchmark.
# --------------------------------------------------------------------------
def run_benchmark(
    *,
    name: str,
    records: list[dict[str, Any]],
    image_loader: Callable[[dict[str, Any]], Image.Image],
    prompt_builder: Callable[[dict[str, Any]], str],
    output_dir: str,
    max_new_tokens: int = 32,
    stop_on_newline: bool = False,
    scorer: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    progress_every: int = 50,
) -> dict[str, Any] | None:
    """Convenience driver: build trainer, generate for each record on this
    rank's shard, gather, and call scorer (rank 0 only).

    Each record must be a dict that ``image_loader`` and ``prompt_builder``
    know how to handle. Each output record is
    ``{"id": rec["id"], "pred": str, "gt": rec.get("gt")}``.
    """
    trainer = build_trainer_from_args()
    runner = EvalRunner(trainer, max_new_tokens=max_new_tokens)
    logger.info(
        f"eval[{name}]: rank {runner.rank}/{runner.world_size}, "
        f"total records = {len(records)}, max_new_tokens={max_new_tokens}"
    )
    my_idx = runner.my_indices(len(records))
    logger.info(f"eval[{name}]: rank {runner.rank} handles {len(my_idx)} records")

    preds: list[dict[str, Any]] = []
    t0 = time.time()
    for n, i in enumerate(my_idx):
        rec = records[i]
        try:
            img = image_loader(rec)
            q = prompt_builder(rec)
            pred = runner.generate(
                img, q,
                max_new_tokens=max_new_tokens,
                stop_on_newline=stop_on_newline,
            )
        except Exception as e:
            pred = f"<ERROR: {type(e).__name__}: {str(e)[:80]}>"
            logger.warning(f"eval[{name}]: rank {runner.rank} record {rec.get('id')} failed: {e!r}")
        preds.append({"id": rec.get("id"), "pred": pred, "gt": rec.get("gt")})
        if (n + 1) % progress_every == 0:
            dt = time.time() - t0
            rate = (n + 1) / dt
            eta = (len(my_idx) - n - 1) / max(rate, 1e-9)
            logger.info(
                f"eval[{name}]: rank {runner.rank} {n+1}/{len(my_idx)} "
                f"({rate:.2f}/s, ETA {eta/60:.1f}min)"
            )

    runner.write_preds(output_dir, preds)
    elapsed = time.time() - t0

    # Write a per-rank "done" sentinel so the post-process aggregator
    # (run AFTER torchrun exits) knows this rank's preds file is complete.
    # We deliberately do NOT call dist.barrier() here — if one rank
    # crashes mid-generation (CUDA OOM, KDA assert), torchrun tears down
    # all other ranks via SIGTERM and rank 0 may never complete its
    # aggregation. With per-rank sentinels + an external aggregator
    # (see ``score_benchmark_from_files`` and ``aggregate_report.py``),
    # the orchestrator can salvage all partial preds.
    sentinel = Path(output_dir) / f"done_rank{runner.rank}"
    with open(sentinel, "w") as f:
        f.write(f"elapsed={elapsed:.3f}\n")

    # Each rank ALSO records its local elapsed for diagnostics.
    if runner.rank == 0:
        # Rank 0 best-effort writes a partial result file with whatever
        # predictions it can see right now. The external aggregator will
        # overwrite this with the final result after torchrun exits.
        try:
            all_preds_now = []
            out = Path(output_dir)
            for r in range(runner.world_size):
                pf = out / f"preds_rank{r}.jsonl"
                if not pf.exists():
                    continue
                with open(pf) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_preds_now.append(json.loads(line))
                            except Exception:
                                pass
            result: dict[str, Any] = {
                "benchmark": name,
                "n_total": len(records),
                "n_scored": len(all_preds_now),
                "elapsed_sec": elapsed,
                "stage": "rank0_partial",
            }
            if scorer is not None:
                try:
                    result.update(scorer(all_preds_now))
                    result["status"] = "PASS"
                except Exception as e:
                    result["status"] = "FAIL"
                    result["error"] = f"{type(e).__name__}: {e}"
            with open(out / "result.json", "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"eval[{name}]: rank 0 partial result {result.get('primary_score')}")
        except Exception as e:
            logger.warning(f"eval[{name}]: rank 0 partial result write failed: {e}")
        return result
    return None


def score_benchmark_from_files(
    *, name: str,
    output_dir: str,
    n_total: int,
    scorer: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    """Post-process aggregator: gather all preds_rank*.jsonl and score.

    Called from the orchestrator AFTER torchrun exits (regardless of exit
    code), so partial results survive a mid-run crash on any rank. The
    elapsed_sec is taken from the slowest done_rank* sentinel.
    """
    out = Path(output_dir)
    preds: list[dict[str, Any]] = []
    n_files = 0
    for p in sorted(out.glob("preds_rank*.jsonl")):
        n_files += 1
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    preds.append(json.loads(line))
                except Exception:
                    pass
    max_elapsed = 0.0
    for s in out.glob("done_rank*"):
        try:
            for line in open(s):
                if line.startswith("elapsed="):
                    max_elapsed = max(max_elapsed, float(line.split("=")[1]))
        except Exception:
            pass
    result: dict[str, Any] = {
        "benchmark": name,
        "n_total": n_total,
        "n_scored": len(preds),
        "n_rank_files": n_files,
        "elapsed_sec": max_elapsed,
        "stage": "final",
    }
    try:
        result.update(scorer(preds))
        if len(preds) == n_total:
            result["status"] = "PASS"
        elif preds:
            result["status"] = "PARTIAL"
        else:
            result["status"] = "FAIL"
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
    with open(out / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
