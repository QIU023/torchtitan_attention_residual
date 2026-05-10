"""Qualitative inference eval for the post-SFT 447M VLM ckpt.

Runs the SGLang Engine on a handful of held-out images (LLaVA-Pretrain
captions, or a user-supplied dir) and prints (image, generation, gold
caption-if-available) tuples. The point is to verify the model
*responds to instructions* after SFT (vs. just continuing in caption
style like the pretrain ckpt).

Distinct from `smoke_vlm_engine.py` — that's a single-image
boot-and-decode smoke; this is N samples with various prompts to
qualitatively assess instruction-following + hallucination.

Usage::

    bash phase11/post_sft_vlm_smoke.sh   # converts SFT ckpt first
    python phase11/eval_vlm_qualitative.py \\
        --model-path phase11/hf_aligned_447m_vlm_sft1200 \\
        --num-images 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent


_PROMPT_VARIANTS = [
    "<image>\nDescribe this image in one short sentence.",
    "<image>\nWhat is in this picture?",
    "<image>\nWhat objects can you identify in the image?",
    "<image>\nProvide a short caption for this image.",
]


def _sample_images(images_root: Path, json_path: Path, n: int):
    """Pull (image_path, gold_caption) pairs from LLaVA-Pretrain JSON.

    Falls back to walking the images directory if the JSON isn't
    available, in which case gold captions are None.
    """
    pairs: list[tuple[Path, str | None]] = []
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        random.seed(42)
        sample = random.sample(data, min(n * 4, len(data)))
        for s in sample:
            img_rel = s.get("image")
            if not img_rel:
                continue
            img_path = images_root / img_rel
            if img_path.exists():
                gold = ""
                for c in s.get("conversations", []):
                    if c.get("from") == "gpt":
                        gold = c.get("value", "").strip()
                        break
                pairs.append((img_path, gold))
                if len(pairs) >= n:
                    break
    if len(pairs) < n:
        # Fallback: walk dir
        for p in images_root.rglob("*.jpg"):
            pairs.append((p, None))
            if len(pairs) >= n:
                break
    return pairs[:n]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path", type=Path,
        default=_WS / "phase11" / "hf_aligned_447m_vlm_sft1200",
    )
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument(
        "--images-root", type=Path,
        default=Path("/workspace/.hf_home/LLaVA-Pretrain"),
    )
    p.add_argument(
        "--json-path", type=Path,
        default=Path("/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"),
    )
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()

    if not args.model_path.exists():
        print(f"ERROR: VLM ckpt not found at {args.model_path}")
        return 1

    samples = _sample_images(args.images_root, args.json_path, args.num_images)
    if not samples:
        print("ERROR: no test images found")
        return 1
    print(f"[eval] sampled {len(samples)} images")

    print(f"[eval] booting SGLang Engine TP={args.tp_size} from {args.model_path}")
    t0 = time.perf_counter()
    import sglang as sgl

    engine = sgl.Engine(
        model_path=str(args.model_path),
        tp_size=args.tp_size,
        dtype="bfloat16",
        attention_backend="flashinfer",
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        disable_cuda_graph=True,
        disable_radix_cache=True,
    )
    print(f"[eval] engine ready in {time.perf_counter() - t0:.1f}s")

    print()
    print("=" * 78)
    correct, total = 0, 0
    total_t = 0.0
    for i, (img_path, gold) in enumerate(samples):
        prompt = _PROMPT_VARIANTS[i % len(_PROMPT_VARIANTS)]
        t = time.perf_counter()
        out = engine.generate(
            prompt=prompt,
            image_data=str(img_path),
            sampling_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": args.max_new_tokens,
            },
        )
        dt = time.perf_counter() - t
        total_t += dt
        gen = (out.get("text") or "").strip().replace("\n", " ")[:200]

        print(f"[{i+1}/{len(samples)}] {img_path.name}  ({dt:.1f}s)")
        print(f"  prompt: {prompt!r}")
        print(f"  gen   : {gen}")
        if gold:
            print(f"  gold  : {gold[:200]}")
        print()
        total += 1

    print("=" * 78)
    print(f"[eval] {total} samples in {total_t:.1f}s "
          f"({total_t/max(total,1):.2f}s/sample avg)")
    print("[eval] DONE ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
