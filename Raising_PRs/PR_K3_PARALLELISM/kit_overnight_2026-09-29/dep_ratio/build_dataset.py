"""Build the DEP ratio dataset (local, not in git) from jackyhate/text-to-image-2M's data_1024_10K shard
(MIT license; 10,000 FLUX-generated 1024 x 1024 JPEGs with prompts).

Every sample stores K_MAX distinct images (img0.jpg .. img{K_MAX-1}.jpg, the original bytes), a short caption
(the first sentence of the first image's prompt, at most 200 characters) and json {"key", "n_images", "src"}.
n_images, drawn with a fixed seed from N_IMAGES_P, is how many of the stored images a sample uses, so the
per-micro-batch vision load varies while the schema stays the same for the Hugging Face webdataset loader.

Usage: python build_dataset.py <data_1024_10K tar> <output dir>
"""
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile

K_MAX = 4
N_IMAGES_P = [0.15, 0.35, 0.25, 0.15, 0.10]
SAMPLES_PER_SHARD = 500
SEED = 0


def add(tar, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


def short_caption(prompt: str) -> str:
    first = re.split(r"(?<=[.!?])\s", prompt.strip(), maxsplit=1)[0]
    return first[:200]


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rng = random.Random(SEED)
    items = {}
    with tarfile.open(src) as tar:
        for m in tar:
            if not m.isfile():
                continue
            key, ext = m.name.rsplit(".", 1)
            items.setdefault(key, {})[ext] = tar.extractfile(m).read()
    keys = sorted(k for k, v in items.items() if "jpg" in v and "json" in v)
    groups = [keys[i : i + K_MAX] for i in range(0, len(keys) - K_MAX + 1, K_MAX)]
    shard, tar, written, counts = 0, None, [], [0] * (K_MAX + 1)
    for i, group in enumerate(groups):
        if i % SAMPLES_PER_SHARD == 0:
            if tar is not None:
                tar.close()
            path = os.path.join(out, f"t2i1024-{shard:04d}.tar")
            tar = tarfile.open(path, "w")
            written.append(path)
            shard += 1
        n = rng.choices(range(K_MAX + 1), weights=N_IMAGES_P)[0]
        counts[n] += 1
        key = f"s{i:06d}"
        prompt = json.loads(items[group[0]]["json"])["prompt"]
        add(tar, f"{key}.txt", short_caption(prompt).encode())
        add(tar, f"{key}.json", json.dumps({"key": key, "n_images": n, "src": group}).encode())
        for j, g in enumerate(group):
            add(tar, f"{key}.img{j}.jpg", items[g]["jpg"])
    tar.close()
    manifest = {"samples": len(groups), "k_max": K_MAX, "n_images_counts": counts, "seed": SEED, "shards": {}}
    for path in written:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        manifest["shards"][os.path.basename(path)] = {"bytes": os.path.getsize(path), "sha256": h.hexdigest()}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}), len(written), "shards")


if __name__ == "__main__":
    main()
