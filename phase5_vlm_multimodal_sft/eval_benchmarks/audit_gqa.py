"""Audit driver for GQA in-tree eval (37.4 claim).

Reuses the SAME trainer-loading + generate path as eval_common.EvalRunner,
but adds:
  * --audit.limit N           : subset size
  * --audit.blind 0|1         : if 1, zero out pixel_values (blind baseline)
  * --audit.dump  PATH        : dump per-sample jsonl with raw+normalized+correct
  * --audit.out   PATH        : write summary json

Scoring uses score_gqa._normalize / _score (the EXACT in-tree scorer) so the
accuracy is computed identically to the cached 0.3737 number.

Single-rank only (CUDA_VISIBLE_DEVICES=1, nproc_per_node=1).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from phase5_vlm_multimodal_sft.eval_benchmarks import eval_common
from phase5_vlm_multimodal_sft.eval_benchmarks import score_gqa
from torchtitan.tools.logging import logger


def _parse_audit_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--audit.limit", dest="limit", type=int, default=500)
    p.add_argument("--audit.blind", dest="blind", type=int, default=0)
    p.add_argument("--audit.noise", dest="noise", type=int, default=0)
    p.add_argument("--audit.dump", dest="dump", default="")
    p.add_argument("--audit.out", dest="out", default="")
    p.add_argument("--audit.shuffle", dest="shuffle", type=int, default=0)
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    aud = _parse_audit_args()

    # Strip the eval.* args that score_gqa.main would consume but we don't,
    # so ConfigManager doesn't choke. We re-implement the limit ourselves.
    # eval_common.build_trainer_from_args parses mm.* + ConfigManager from
    # sys.argv, so leave those intact. Remove --eval.* tokens.
    cleaned = []
    skip = False
    for tok in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if tok.startswith("--eval."):
            # value follows
            skip = True
            continue
        cleaned.append(tok)
    sys.argv = [sys.argv[0]] + cleaned

    # ---- Load records + images via the in-tree loaders ----
    if aud.shuffle:
        # Load ALL records, deterministically shuffle, then take the slice.
        records, imgs = score_gqa._load_records(limit=None)
        import random
        random.Random(aud.shuffle).shuffle(records)
        if aud.limit:
            records = records[:aud.limit]
    else:
        records, imgs = score_gqa._load_records(limit=aud.limit or None)
    score_gqa._REC_IMG_CACHE = imgs
    logger.info(
        f"audit: loaded {len(records)} records, blind={aud.blind} "
        f"noise={aud.noise} shuffle={aud.shuffle}"
    )

    # ---- Build trainer + runner (identical path to run_benchmark) ----
    trainer = eval_common.build_trainer_from_args()
    runner = eval_common.EvalRunner(trainer, max_new_tokens=16)

    blind = bool(aud.blind)
    noise = bool(aud.noise)

    # ---- Monkeypatch generate to (optionally) zero pixel_values and to
    #      expose the raw decoded text. We replicate the body of
    #      EvalRunner.generate exactly, changing only the pixel tensor. ----
    @torch.no_grad()
    def generate_audited(image, question, max_new_tokens=16, stop_on_newline=True):
        prompt_ids = runner.build_input_ids(question)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=runner.device)
        pixel = runner.image_processor(
            images=image.convert("RGB"), return_tensors="pt",
        )["pixel_values"].to(device=runner.device, dtype=torch.bfloat16)
        if blind:
            pixel = torch.zeros_like(pixel)  # <-- BLIND: drop all visual info
        elif noise:
            pixel = torch.randn_like(pixel)  # <-- NOISE: random image, no real content
        vision_features = runner.vision_tower(pixel_values=pixel).last_hidden_state
        vision_embeds = runner.projector(vision_features)
        generated = []
        for _ in range(max_new_tokens):
            cur_ids = torch.cat([
                input_ids,
                torch.tensor([generated], dtype=torch.long, device=runner.device)
                if generated else torch.empty((1, 0), dtype=torch.long, device=runner.device),
            ], dim=1)
            cur_mask = (cur_ids == runner.image_sentinel)
            logits = runner.lm(cur_ids, vision_embeds=vision_embeds, image_mask=cur_mask)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == runner.eos_id:
                break
            if stop_on_newline and runner.newline_id is not None and next_id == runner.newline_id:
                break
            generated.append(next_id)
        text = runner.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    # ---- Hot loop ----
    dump_rows = []
    preds = []
    t0 = time.time()
    for n, rec in enumerate(records):
        try:
            img = score_gqa._image_loader(rec)
            q = score_gqa._prompt_builder(rec)
            raw = generate_audited(img, q, max_new_tokens=16, stop_on_newline=True)
        except Exception as e:
            raw = f"<ERROR: {type(e).__name__}: {str(e)[:80]}>"
            logger.warning(f"audit: rec {rec.get('id')} failed: {e!r}")
        gt = rec["gt"]
        npred = score_gqa._normalize(raw)
        ngt = score_gqa._normalize(gt)
        correct = (npred == ngt)
        preds.append({"id": rec["id"], "pred": raw, "gt": gt})
        dump_rows.append({
            "id": rec["id"],
            "question": rec["question"],
            "gold": gt,
            "raw_pred": raw,
            "norm_pred": npred,
            "norm_gold": ngt,
            "correct": correct,
            "structural": rec.get("structural"),
        })
        if (n + 1) % 100 == 0:
            dt = time.time() - t0
            rate = (n + 1) / dt
            logger.info(f"audit: {n+1}/{len(records)} ({rate:.2f}/s)")

    # ---- Score with the EXACT in-tree scorer ----
    result = score_gqa._score(preds)
    result["blind"] = blind
    result["noise"] = noise
    result["n_records"] = len(records)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    logger.info(f"audit RESULT: {json.dumps(result)}")

    if aud.dump:
        with open(aud.dump, "w") as f:
            for r in dump_rows:
                f.write(json.dumps(r) + "\n")
        logger.info(f"audit: dumped {len(dump_rows)} rows → {aud.dump}")
    if aud.out:
        with open(aud.out, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"audit: wrote summary → {aud.out}")

    print("AUDIT_SUMMARY " + json.dumps(result))


if __name__ == "__main__":
    main()
