#!/usr/bin/env python3
"""Build a LlavaCaptionTask-format json from COCO val2017 captions, written to the
GRPO default path. Real (image, human-caption) pairs for substantive captioning RL.
images_dir stays /workspace/.hf_home/LLaVA-Pretrain (config default); we symlink
val2017/ into it so image relpaths resolve."""
import json, os, random

COCO = "/workspace/coco_rl"
LLAVA = "/workspace/.hf_home/LLaVA-Pretrain"
N = int(os.environ.get("N_SAMPLES", "5000"))

# symlink COCO images into the GRPO images_dir
os.makedirs(LLAVA, exist_ok=True)
link = os.path.join(LLAVA, "val2017")
if not os.path.exists(link):
    os.symlink(os.path.join(COCO, "val2017"), link)

cap = json.load(open(os.path.join(COCO, "annotations/captions_val2017.json")))
id2file = {im["id"]: im["file_name"] for im in cap["images"]}
# one caption per image (first), keep only images that exist
by_img = {}
for a in cap["annotations"]:
    iid = a["image_id"]
    if iid in by_img:
        continue
    fn = id2file.get(iid)
    if not fn:
        continue
    cap_txt = a["caption"].strip()
    if not cap_txt.endswith("."):
        cap_txt += "."
    cap_txt = cap_txt[0].upper() + cap_txt[1:] if cap_txt else cap_txt
    by_img[iid] = (fn, cap_txt)

items = list(by_img.values())
random.Random(42).shuffle(items)
items = items[:N]
records = [{
    "id": f"coco{i}",
    "image": f"val2017/{fn}",
    "conversations": [
        {"from": "human", "value": "<image>\nDescribe the image briefly."},
        {"from": "gpt", "value": cap},
    ],
} for i, (fn, cap) in enumerate(items)]

out = os.path.join(LLAVA, "blip_laion_cc_sbu_558k.json")
json.dump(records, open(out, "w"))
print(f"wrote {len(records)} COCO caption records -> {out}")
print(f"images via symlink {link} -> {os.path.realpath(link)}")
print("sample:", records[0]["image"], "|", records[0]["conversations"][1]["value"][:70])
