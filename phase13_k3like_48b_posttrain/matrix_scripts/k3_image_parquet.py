"""A small image GRPO set for the Kimi K3 engine cells: solid-colour 56x56 images, the prompt asks
for the colour in one word, the reward is the colour name. Schema on the shape of geo3k."""
import argparse
import io
import random

import datasets
from PIL import Image

COLOURS = {
    "red": (220, 30, 30),
    "green": (30, 200, 40),
    "blue": (30, 60, 230),
    "yellow": (240, 220, 40),
    "white": (250, 250, 250),
    "black": (5, 5, 5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/.k3_image_data/train.parquet")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--size", type=int, default=56)
    args = ap.parse_args()
    rng = random.Random(0)
    rows = []
    names = list(COLOURS)
    for i in range(args.rows):
        name = names[i % len(names)]
        base = COLOURS[name]
        colour = tuple(min(255, max(0, c + rng.randint(-5, 5))) for c in base)
        buf = io.BytesIO()
        Image.new("RGB", (args.size, args.size), colour).save(buf, format="PNG")
        rows.append(
            {
                "data_source": "k3_colour",
                "prompt": [
                    {"role": "user", "content": "<image>What colour is this image? Answer with one word."}
                ],
                "images": [{"bytes": buf.getvalue(), "path": None}],
                "ability": "vision",
                "reward_model": {"style": "rule", "ground_truth": name},
                "extra_info": {"index": i, "answer": name},
            }
        )
    ds = datasets.Dataset.from_list(rows)
    ds = ds.cast_column("images", datasets.Sequence(datasets.Image()))
    ds.to_parquet(args.out)
    print(args.out, len(ds), ds.features)


if __name__ == "__main__":
    main()
