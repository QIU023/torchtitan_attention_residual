"""CPU-only diagnostics for the step-~2655 KDA device-side-assert crash.

The full SFT run (phase5/runs/mm_sft_447m_full) trained healthily for
~2650 steps then died with a CUDA device-side assert that *surfaced*
inside fla's chunk_kda (`final_state = k.new_zeros(...)` in
fla/ops/common/chunk_delta_h.py). `new_zeros` cannot device-assert; the
real failing kernel ran earlier and async.

This script does the CPU-side half of the investigation:
  1. Scans the LLaVA-Pretrain JSON for degenerate records.
  2. Scans the images each dp-rank would consume around the crash step
     for corruption / truncation / degenerate dimensions.
  3. Reconstructs the exact records dp_rank=7 (the rank that died) would
     have consumed near the crash step, so a GPU repro can feed just
     those samples.

It does NOT touch the GPU. Run:  /usr/bin/python3 phase5/diagnose_kda_crash.py
"""
from __future__ import annotations

import json
import os
import warnings

from PIL import Image

JSON = "/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
IMGDIR = "/workspace/.hf_home/LLaVA-Pretrain"

# Run geometry (from runs/mm_sft_447m_full/train.log):
#   dp_world_size = 8, local_bs = 4, grad_accum = 2  -> 8 samples/rank/step
#   num_workers = 0  -> rank r iterates records range(r, N, 8)
#   crash on rank 7, between logged step 2650 and 2660.
DP_WORLD = 8
SAMPLES_PER_RANK_PER_STEP = 8
CRASH_RANK = 7
CRASH_STEP_LO, CRASH_STEP_HI = 2651, 2660


def scan_json_degenerate(recs: list) -> None:
    empty = no_conv = no_image = 0
    for r in recs:
        convs = r.get("conversations", [])
        cap = next(
            (t.get("value", "") for t in convs if t.get("from") == "gpt"), ""
        )
        no_conv += not convs
        empty += not cap.strip()
        no_image += "image" not in r
    print(
        f"[json] N={len(recs):,}  no_conversations={no_conv}  "
        f"empty_caption={empty}  no_image_field={no_image}"
    )


def scan_images(recs: list, lo: int, hi: int, stride: int) -> None:
    bad, tiny, checked = [], [], 0
    for i in range(lo, hi, stride):
        if i >= len(recs):
            break
        ip = os.path.join(IMGDIR, recs[i]["image"])
        if not os.path.isfile(ip):
            continue
        checked += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # truncation warnings -> errors
                im = Image.open(ip)
                im.load()
                im = im.convert("RGB")
            w, h = im.size
            if w < 8 or h < 8:
                tiny.append((i, recs[i]["image"], (w, h)))
        except Exception as e:  # noqa: BLE001
            bad.append((i, recs[i]["image"], type(e).__name__, str(e)[:80]))
    print(
        f"[images {lo}-{hi} step {stride}] checked={checked}  "
        f"corrupt={len(bad)}  tiny(<8px)={len(tiny)}"
    )
    for b in bad[:20]:
        print("  CORRUPT", b)
    for t in tiny[:20]:
        print("  TINY", t)


def crash_record_window(n_records: int) -> tuple[int, int]:
    """Records dp_rank=7 had consumed by the crash step.

    rank 7 iterates range(7, N, 8); it consumes 8 samples/step. After S
    steps it has pulled ~8*S samples, i.e. record index ~ 7 + 8*S*8.
    Records with a missing image file are skipped (`continue`) so the
    record cursor runs slightly ahead of consumed-sample count. We widen
    the window generously and also account for the dataloader prefetch.
    """
    lo_step, hi_step = CRASH_STEP_LO, CRASH_STEP_HI
    # idx ~= rank + DP_WORLD * (samples_consumed); samples_consumed ~= 8*step
    lo = CRASH_RANK + DP_WORLD * (SAMPLES_PER_RANK_PER_STEP * lo_step)
    hi = CRASH_RANK + DP_WORLD * (SAMPLES_PER_RANK_PER_STEP * hi_step)
    # widen by +/- 4000 records for skip-ahead + prefetch slack
    return max(0, lo - 4000), min(n_records, hi + 4000)


def main() -> None:
    recs = json.load(open(JSON))
    scan_json_degenerate(recs)

    lo, hi = crash_record_window(len(recs))
    print(f"[crash] rank {CRASH_RANK} record window estimate: {lo}-{hi}")
    # dense scan of the crash window
    scan_images(recs, lo, hi, stride=1)
    # sparse scan of the whole second half (catch any global corruption)
    scan_images(recs, len(recs) // 2, len(recs), stride=37)


if __name__ == "__main__":
    main()
