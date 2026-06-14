"""Unified audit driver for in-tree VLM eval with the CORRECT trained projector.

Root-cause of the "no visual grounding" result: ``eval_common.load_ckpt_only``
calls ``checkpointer.load(step=0)`` with ``initial_load_model_only=True``, and
``Checkpointer._states_to_load(model_only=True)`` returns ONLY
``self.states[MODEL].state_dict()`` (torchtitan/components/checkpoint.py:787-788).
The trained projector lives in the separate ``mm_state`` checkpointer entry
(``mm_state.projector.{fc1,fc2}.{weight,bias}``), so it is NEVER loaded — eval
silently runs the RANDOM-init projector (fc{1,2}.bias == 0, weight ~trunc_normal).

This script fixes that the fast way (path b): after the normal model-only load
restores the LM, it MANUALLY injects the trained projector tensors (read from
the HF export ``popefix_clean/model.safetensors`` under keys
``mm_projector.projector.{fc1,fc2}.{weight,bias}``, which is cos=0.9999 to the
DCP ``mm_state.projector``) into ``trainer.projector``, overwriting the random
init. It then runs GQA or POPE real-vs-blind exactly like ``audit_gqa.py``.

Usage (single rank, CUDA_VISIBLE_DEVICES=1):
  --audit.bench   gqa|pope
  --audit.limit   N
  --audit.blind   0|1     (1 -> pixel_values = zeros_like, the blind baseline)
  --audit.shuffle SEED    (0 = head slice; >0 = deterministic shuffle then slice)
  --audit.inject  1|0     (1 = inject trained projector [default]; 0 = leave random)
  --audit.proj_src PATH   (HF safetensors with mm_projector.* ; default popefix_clean)
  --audit.out     PATH
  --audit.dump    PATH
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from phase5_vlm_multimodal_sft.eval_benchmarks import eval_common  # noqa: E402
from phase5_vlm_multimodal_sft.eval_benchmarks import score_gqa  # noqa: E402
from phase5_vlm_multimodal_sft.eval_benchmarks import score_pope  # noqa: E402
from torchtitan.tools.logging import logger  # noqa: E402


DEFAULT_PROJ_SRC = (
    "/home/torchtitan_attention_residual/phase11_rlhf_grpo_infra/hf/"
    "popefix_clean/model.safetensors"
)
# HF export key prefix -> our Projector submodule param name.
_HF_KEYS = {
    "fc1.weight": "mm_projector.projector.fc1.weight",
    "fc1.bias": "mm_projector.projector.fc1.bias",
    "fc2.weight": "mm_projector.projector.fc2.weight",
    "fc2.bias": "mm_projector.projector.fc2.bias",
}


def _parse_audit_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--audit.bench", dest="bench", default="gqa")
    p.add_argument("--audit.limit", dest="limit", type=int, default=500)
    p.add_argument("--audit.blind", dest="blind", type=int, default=0)
    p.add_argument("--audit.shuffle", dest="shuffle", type=int, default=0)
    p.add_argument("--audit.inject", dest="inject", type=int, default=1)
    p.add_argument("--audit.proj_src", dest="proj_src", default=DEFAULT_PROJ_SRC)
    p.add_argument("--audit.out", dest="out", default="")
    p.add_argument("--audit.dump", dest="dump", default="")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _read_hf_projector(path: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as st:
        for local, hf_key in _HF_KEYS.items():
            out[local] = st.get_tensor(hf_key)
    return out


def _module_param(projector, dotted: str) -> torch.Tensor:
    """Return the live nn.Parameter tensor for e.g. 'fc1.weight'."""
    obj = projector
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return getattr(obj, parts[-1])


def _to_plain(t: torch.Tensor) -> torch.Tensor:
    """DTensor -> full local tensor (dp=1 -> Replicate, so full == local)."""
    if hasattr(t, "full_tensor"):
        try:
            return t.full_tensor()
        except Exception:
            pass
    return t


@torch.no_grad()
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = _to_plain(a).detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


@torch.no_grad()
def _inject_projector(projector, ref: dict[str, torch.Tensor]) -> dict:
    """Overwrite the projector's params with the trained tensors, in place.
    Returns a verification report (pre/post cos to ref, bias-nonzero flags)."""
    report = {"pre_cos": {}, "post_cos": {}, "bias_nonzero": {},
              "bias_absmax": {}}
    # --- pre-injection cos (random init vs trained) ---
    for local in _HF_KEYS:
        live = _module_param(projector, local)
        report["pre_cos"][local] = _cos(live, ref[local])

    # --- inject: copy_ into the live param (handles DTensor by writing the
    #     local shard; dp=1 => Replicate => local shard == full tensor) ---
    for local in _HF_KEYS:
        live = _module_param(projector, local)
        src = ref[local].to(dtype=live.dtype if not hasattr(live, "dtype")
                             else _to_plain(live).dtype)
        if hasattr(live, "to_local"):
            local_t = live.to_local()
            local_t.copy_(src.to(device=local_t.device, dtype=local_t.dtype))
        else:
            live.copy_(src.to(device=live.device, dtype=live.dtype))

    # --- post-injection cos + bias diagnostics ---
    for local in _HF_KEYS:
        live = _module_param(projector, local)
        report["post_cos"][local] = _cos(live, ref[local])
        if local.endswith(".bias"):
            v = _to_plain(live).detach().float()
            report["bias_nonzero"][local] = bool(v.abs().sum().item() > 0)
            report["bias_absmax"][local] = float(v.abs().max().item())
    return report


@torch.no_grad()
def _bias_state(projector) -> dict:
    """Report current bias absmax (to show random-init bias == 0)."""
    out = {}
    for local in ("fc1.bias", "fc2.bias"):
        v = _to_plain(_module_param(projector, local)).detach().float()
        out[local] = float(v.abs().max().item())
    return out


def main():
    aud = _parse_audit_args()

    # Strip --eval.* tokens so ConfigManager doesn't choke (mirrors audit_gqa).
    cleaned, skip = [], False
    for tok in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if tok.startswith("--eval."):
            skip = True
            continue
        cleaned.append(tok)
    sys.argv = [sys.argv[0]] + cleaned

    bench = aud.bench.lower()
    assert bench in ("gqa", "pope"), f"unknown bench {bench!r}"

    # ---- Build trainer (normal model-only load: LM correct, projector RANDOM)
    trainer = eval_common.build_trainer_from_args()
    runner = eval_common.EvalRunner(trainer, max_new_tokens=16)

    # ---- projector load verification + (optional) injection ----
    pre_bias = _bias_state(runner.projector)
    logger.info(f"audit: projector bias BEFORE inject (random-init expects ~0): {pre_bias}")
    if aud.inject:
        ref = _read_hf_projector(aud.proj_src)
        rep = _inject_projector(runner.projector, ref)
        logger.info("PROJ_VERIFY " + json.dumps(rep))
    else:
        logger.info("audit: inject=0, running with RANDOM projector (legacy bug repro)")

    # ---- load records ----
    if bench == "gqa":
        if aud.shuffle:
            records, imgs = score_gqa._load_records(limit=None)
            import random
            random.Random(aud.shuffle).shuffle(records)
            if aud.limit:
                records = records[:aud.limit]
        else:
            records, imgs = score_gqa._load_records(limit=aud.limit or None)
        score_gqa._REC_IMG_CACHE = imgs
        image_loader = score_gqa._image_loader
        prompt_builder = score_gqa._prompt_builder
        scorer = score_gqa._score
        max_new = 16
    else:  # pope
        records = score_pope._load_records(limit=aud.limit or None)
        if aud.shuffle:
            import random
            random.Random(aud.shuffle).shuffle(records)
        image_loader = score_pope._image_loader
        prompt_builder = score_pope._prompt_builder
        scorer = score_pope._score
        max_new = 12

    blind = bool(aud.blind)
    logger.info(
        f"audit: bench={bench} loaded {len(records)} records blind={blind} "
        f"inject={aud.inject} shuffle={aud.shuffle}"
    )

    @torch.no_grad()
    def generate_audited(image, question, max_new_tokens, stop_on_newline=True):
        prompt_ids = runner.build_input_ids(question)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=runner.device)
        pixel = runner.image_processor(
            images=image.convert("RGB"), return_tensors="pt",
        )["pixel_values"].to(device=runner.device, dtype=torch.bfloat16)
        if blind:
            pixel = torch.zeros_like(pixel)
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
        return runner.tokenizer.decode(generated, skip_special_tokens=True).strip()

    preds, dump_rows = [], []
    t0 = time.time()
    for n, rec in enumerate(records):
        try:
            img = image_loader(rec)
            q = prompt_builder(rec)
            raw = generate_audited(img, q, max_new, stop_on_newline=True)
        except Exception as e:
            raw = f"<ERROR: {type(e).__name__}: {str(e)[:80]}>"
            logger.warning(f"audit: rec {rec.get('id')} failed: {e!r}")
        preds.append({"id": rec["id"], "pred": raw, "gt": rec["gt"]})
        dump_rows.append({"id": rec["id"], "question": rec.get("question"),
                          "gold": rec["gt"], "raw_pred": raw})
        if (n + 1) % 100 == 0:
            rate = (n + 1) / (time.time() - t0)
            logger.info(f"audit: {n+1}/{len(records)} ({rate:.2f}/s)")

    result = scorer(preds)
    result["bench"] = bench
    result["blind"] = blind
    result["inject"] = bool(aud.inject)
    result["shuffle"] = aud.shuffle
    result["n_records"] = len(records)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    logger.info(f"audit RESULT: {json.dumps(result)}")

    if aud.dump:
        with open(aud.dump, "w") as f:
            for r in dump_rows:
                f.write(json.dumps(r) + "\n")
    if aud.out:
        with open(aud.out, "w") as f:
            json.dump(result, f, indent=2)

    print("AUDIT_SUMMARY " + json.dumps(result))


if __name__ == "__main__":
    main()
