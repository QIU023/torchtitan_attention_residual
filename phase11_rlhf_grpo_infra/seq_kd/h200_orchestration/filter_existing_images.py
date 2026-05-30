"""Drop conversations whose image file does not exist under IMAGE_ROOT.

seq-KD distilled JSON keeps mix665k's image references (coco/gqa/ocr_vqa/
textvqa/vg). If a source isn't downloaded, the student dataloader would crash.
Pre-filter to rows whose image is present (or text-only rows with no image).
"""
import json, os, sys

src, dst, image_root = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(src))
kept, dropped = [], 0
for r in data:
    img = r.get("image")
    if img is None:
        kept.append(r); continue
    if os.path.exists(os.path.join(image_root, img)):
        kept.append(r)
    else:
        dropped += 1
json.dump(kept, open(dst, "w"), ensure_ascii=False)
print(f"FILTER kept={len(kept)} dropped={dropped} -> {dst}")
