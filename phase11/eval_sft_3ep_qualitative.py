"""Qualitative eval for the SFT 3ep VLM ckpt.

Picks 10 random LLaVA-Pretrain images, runs the SGLang VLM engine on
them with both T=0 greedy and T=0.7 sampling, prints the generation +
checks for the EOS-trap pattern (output collapses to `!!!!` or starts
with non-letters indicating the model didn't attend to the image).

Exits with code 0 if **at least 6/10 samples produce coherent text
that starts with a letter and has < 30% '!' density**; nonzero
otherwise. This is the gate before launching the GRPO stage.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path


def is_garbage(text: str) -> tuple[bool, str]:
    """Return (is_garbage, reason)."""
    stripped = text.strip()
    if not stripped:
        return True, "empty"
    if not re.match(r"^[A-Za-z\"]", stripped):
        return True, f"starts non-letter ({stripped[:5]!r})"
    bang_count = stripped.count("!")
    if bang_count / max(len(stripped), 1) > 0.3:
        return True, f"bang_density={bang_count}/{len(stripped)}"
    return False, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--images-dir", type=Path,
                   default=Path("/workspace/.hf_home/LLaVA-Pretrain"))
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--gate-threshold", type=float, default=0.6,
                   help="Min fraction of samples that must be 'coherent'")
    args = p.parse_args()

    # Sample images
    rng = random.Random(42)
    all_images = []
    for bucket in args.images_dir.glob("0000*/*.jpg"):
        all_images.append(bucket)
        if len(all_images) > 500:
            break
    if not all_images:
        print(f"[eval] no images under {args.images_dir}/00000*/")
        return 2
    selected = rng.sample(all_images, args.num_samples)
    print(f"[eval] sampled {len(selected)} images")

    from sglang.srt.entrypoints.engine import Engine
    print(f"[eval] booting SGLang Engine on {args.model_path}")
    t0 = time.perf_counter()
    e = Engine(
        model_path=str(args.model_path),
        tp_size=1,
        dtype="bfloat16",
        attention_backend="flashinfer",
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        disable_radix_cache=True,
    )
    print(f"[eval] engine ready in {time.perf_counter()-t0:.1f}s")

    coherent = 0
    fail_reasons: list[str] = []
    for i, img in enumerate(selected):
        prompt = (
            "You are a helpful vision assistant. Describe the image in "
            "one short sentence.\n\n<image>\nUser: Describe the image.\n"
            "Assistant:"
        )
        # Greedy
        out = e.generate(
            prompt=prompt, image_data=str(img),
            sampling_params={"temperature": 0.0, "max_new_tokens": 50, "stop": []},
        )
        text_greedy = out.get("text", "").strip()
        garbage_g, reason_g = is_garbage(text_greedy)
        # Sampling
        out2 = e.generate(
            prompt=prompt, image_data=str(img),
            sampling_params={"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 50, "stop": []},
        )
        text_sample = out2.get("text", "").strip()
        garbage_s, reason_s = is_garbage(text_sample)

        # Count as coherent if EITHER temperature produces non-garbage
        sample_coherent = (not garbage_g) or (not garbage_s)
        if sample_coherent:
            coherent += 1
            mark = "+"
        else:
            mark = "-"
            fail_reasons.append(f"img{i} greedy={reason_g} sample={reason_s}")
        print(f"[eval] {mark} img={img.name}")
        print(f"       T=0:   {text_greedy[:150]!r}")
        print(f"       T=0.7: {text_sample[:150]!r}")

    rate = coherent / args.num_samples
    print(f"\n[eval] coherent {coherent}/{args.num_samples} ({rate*100:.0f}%)")
    print(f"[eval] gate threshold: {args.gate_threshold*100:.0f}%")
    if fail_reasons:
        print("[eval] failures:")
        for r in fail_reasons:
            print(f"  {r}")
    e.shutdown()
    return 0 if rate >= args.gate_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
