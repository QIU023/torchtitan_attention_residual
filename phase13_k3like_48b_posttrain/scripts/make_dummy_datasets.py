"""Generate dummy SFT and GRPO datasets matching the chosen real ones.

Same field names and collator contract as LLaVA-Instruct-150K, OpenHermes-2.5,
GSM8K and VQAv2, at a size that fits one 16 GB GPU. This exercises wiring only.
A smoke that passes here proves the plumbing and proves nothing about quality --
label any curve produced from it accordingly.

Usage: python make_dummy_datasets.py --out /workspace/dummy_data [--n 256]
"""

from __future__ import annotations

import argparse
import json
import os
import random


def _sentence(rng: random.Random, n: int) -> str:
    words = ["the", "model", "attends", "over", "blocks", "and", "routes",
             "tokens", "to", "experts", "before", "the", "residual", "add"]
    return " ".join(rng.choice(words) for _ in range(n))


def gsm8k(rng: random.Random, n: int) -> list[dict]:
    """GSM8K: question + answer whose final line is '#### <number>'.

    The reward function extracts that number and exact-matches it, so the
    dummy has to reproduce the marker exactly or the reward path is untested.
    """
    out = []
    for _ in range(n):
        a, b = rng.randint(2, 40), rng.randint(2, 40)
        out.append({
            "question": f"A has {a} items and B has {b} more. How many in total?",
            "answer": f"B has {a} + {b} = {a + b}.\n#### {a + b}",
        })
    return out


def vqav2(rng: random.Random, n: int, img_px: int) -> list[dict]:
    """VQAv2: question + ten annotator answers; reward is VQA accuracy."""
    out = []
    colours = ["red", "blue", "green", "yellow"]
    for i in range(n):
        gt = rng.choice(colours)
        # nine agreeing annotators and one dissenting, so accuracy is not
        # trivially 1.0 and the min(count/3, 1) rule actually gets exercised
        answers = [gt] * 9 + [rng.choice([c for c in colours if c != gt])]
        out.append({
            "question_id": i,
            "question": "What colour is the object?",
            "image": f"images/{i:06d}.png",
            "answers": answers,
            "multiple_choice_answer": gt,
        })
    return out


def llava(rng: random.Random, n: int) -> list[dict]:
    """LLaVA-Instruct: conversations with an <image> token in the human turn."""
    return [{
        "id": f"dummy-{i:06d}",
        "image": f"images/{i:06d}.png",
        "conversations": [
            {"from": "human", "value": f"<image>\n{_sentence(rng, 12)}?"},
            {"from": "gpt", "value": _sentence(rng, 24)},
        ],
    } for i in range(n)]


def hermes(rng: random.Random, n: int) -> list[dict]:
    """OpenHermes-2.5: system/human/gpt conversation triples, text only."""
    return [{
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": f"{_sentence(rng, 14)}?"},
            {"from": "gpt", "value": _sentence(rng, 30)},
        ],
    } for i in range(n)]


def write_images(path: str, n: int, px: int) -> None:
    """Solid-colour PNGs. Content is irrelevant; the preprocessing path is not
    -- NaViT resize and packing must see real files with real dimensions."""
    from PIL import Image

    os.makedirs(path, exist_ok=True)
    rng = random.Random(0)
    for i in range(n):
        c = tuple(rng.randint(0, 255) for _ in range(3))
        # vary the aspect ratio so navit_resize is actually exercised
        w = px + rng.choice([0, px // 2])
        Image.new("RGB", (w, px), c).save(os.path.join(path, f"{i:06d}.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/dummy_data")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--image-px", type=int, default=112)
    args = ap.parse_args()

    rng = random.Random(42)
    sets = {
        "gsm8k_dummy": gsm8k(rng, args.n),
        "vqav2_dummy": vqav2(rng, args.n, args.image_px),
        "llava_instruct_dummy": llava(rng, args.n),
        "openhermes_dummy": hermes(rng, args.n),
    }
    for name, rows in sets.items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "data.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows -> {d}/data.jsonl")

    for name in ("vqav2_dummy", "llava_instruct_dummy"):
        write_images(os.path.join(args.out, name, "images"), args.n,
                     args.image_px)
        print(f"{name}: {args.n} images -> {args.out}/{name}/images")


if __name__ == "__main__":
    main()
