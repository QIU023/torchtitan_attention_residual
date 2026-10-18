"""Local kit (never committed): titan's K3 video chunks against the released Kimi K2.5 processor code, run with K3's
preprocessor config. Part A sweeps (frame count, fps) through both chunking rules on stubbed metadata and compares the
sampled frame ids and the chunk prompt text. Part B decodes real videos once with PyAV, feeds the same frames to
titan's chunk function and to K2.5's split_video_chunks + preprocess, and compares ids, prompts, grids and pixels."""
import json
import random
import re
import sys
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import k25ref.kimi_k25_vision_processing as kvp
from k25ref import media_utils as mu
from k25ref.kimi_k25_vision_processing import KimiK25VisionProcessor
from torchtitan.components.data.multimodal import mm_datasets
from torchtitan.components.data.multimodal.utils.image import resize_to_navit_patch_grid, vision_to_patches
from torchtitan.components.data.multimodal.utils.video import load_video_frames, sample_frame_indices

CFG = json.load(open("/workspace/k3qat_mm_hf/preprocessor_config.json"))["media_proc_cfg"]
PROC = KimiK25VisionProcessor(media_proc_cfg=CFG)
HEADER = "video<|media_content|>"
PAD = re.compile(r"(<\|media_pad\|>)+")


class _Tok:
    video_token = image_token = "<|media_pad|>"
    vision_start_token, vision_end_token, eos_token = "<|media_begin|>", "<|media_end|>", "[EOS]"


def k25_chunks(num_frames, fps, frames=None):
    """K2.5's split_video_chunks on stubbed metadata; frames (a list of PIL images by source index) when given."""
    got = {}
    kvp.get_video_meta = lambda url: mu.VideoSpec(media_type="video", height=1, width=1, num_frames=num_frames, fps=fps)

    def resampling(url, ids, **kw):
        got["ids"] = list(ids)
        return [frames[i] if frames is not None else Image.new("RGB", (2, 2)) for i in ids]

    kvp.resampling = resampling
    chunks = PROC.split_video_chunks(b"stub")
    return got["ids"], chunks


def titan_sample(frames_THWC, times, *, real_resize):
    captured = {}
    original = mm_datasets._tokenize_mm_text

    def capture(text, tokenizer):
        captured["text"] = text
        return {"input_ids": torch.zeros(1, dtype=torch.long)}

    kwargs = dict(
        text="CAPTION", frames=frames_THWC, tokenizer=_Tok(), patch_size=14, temporal_patch_size=1, spatial_merge_size=2,
        min_pixels=1, max_pixels=1 << 40, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
        resize_fn=resize_to_navit_patch_grid, max_patches=CFG["in_patch_limit_each_frame"],
        max_patches_per_side=CFG["patch_limit_on_one_side"], frame_times=times,
        frames_per_chunk=CFG["temporal_merge_kernel_size"], chunk_header=HEADER,
    )
    with patch.object(mm_datasets, "_tokenize_mm_text", capture):
        if real_resize:
            sample = mm_datasets._process_chunked_video_text_sample(**kwargs)
        else:
            with patch.object(mm_datasets, "process_video", lambda video, **kw: video.float()):
                sample = mm_datasets._process_chunked_video_text_sample(**kwargs)
    prompts = PAD.sub("<|media_pad|>", captured["text"]).split("CAPTION")[0]
    return sample, prompts


def part_a():
    fps_list = [1.0, 5.0, 7.5, 8.0, 10.0, 12.0, 15.0, 24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 50.0, 60000 / 1001,
                60.0, 120.0]
    rng = random.Random(0)
    fps_list += [rng.uniform(0.5, 240.0) for _ in range(60)]
    counts = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 299, 300, 301, 1000, 1801, 3600, 9000, 18000]
    counts += [rng.randint(1, 20000) for _ in range(15)]
    cases = ids_bad = prompt_bad = 0
    for fps in fps_list:
        for n in counts:
            ref_ids, chunks = k25_chunks(n, fps)
            ids = sample_frame_indices(n, fps, CFG["sample_fps"])
            frames = torch.zeros(len(ids), 28, 28, 3, dtype=torch.uint8)
            _, prompts = titan_sample(frames, [i / fps for i in ids], real_resize=False)
            ref_prompts = "".join(c["prompt"] for c in chunks)
            cases += 1
            if ids != ref_ids:
                ids_bad += 1
                if ids_bad <= 3:
                    print(f"  ids differ: n={n} fps={fps}")
            if prompts != ref_prompts:
                prompt_bad += 1
                if prompt_bad <= 3:
                    print(f"  prompts differ: n={n} fps={fps}\n    titan {prompts[:200]}\n    k25   {ref_prompts[:200]}")
    print(f"part A: {cases} (frame count, fps) cases, frame ids differ in {ids_bad}, chunk prompts differ in {prompt_bad}")


def part_b(paths):
    import av

    for path in paths:
        with av.open(path) as container:
            stream = container.streams.video[0]
            n, fps = stream.frames, float(stream.average_rate)
            container.seek(0)
            decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        loaded = load_video_frames(path, CFG["sample_fps"])
        assert loaded is not None
        frames, times = loaded
        ref_ids, chunks = k25_chunks(n, fps, frames=[Image.fromarray(f) for f in decoded])
        ids = [round(t * fps) for t in times]
        same_frames = all(np.array_equal(frames[k].numpy(), decoded[i]) for k, i in enumerate(ref_ids)) if ids == ref_ids else False
        sample, prompts = titan_sample(frames, times, real_resize=True)
        ref = PROC.preprocess(chunks, return_tensors="pt")
        patches, grids = zip(*(vision_to_patches(c, 14, 1, 2, patch_order="raster") for c in sample["pixel_values"]))
        titan_pixels, titan_grids = torch.cat(patches), torch.stack(grids)
        ref_pixels = ref["pixel_values"].flatten(1).float()
        diff = (titan_pixels.float() - ref_pixels).abs() if titan_pixels.shape == ref_pixels.shape else None
        print(f"part B {path.split('/')[-1]}: {n} frames at {fps:.4f} fps, sampled {len(ref_ids)} in {len(chunks)} chunks; "
              f"ids equal {ids == ref_ids}, decoded frames equal {same_frames}, prompts equal {prompts == ''.join(c['prompt'] for c in chunks)}, "
              f"grids equal {torch.equal(titan_grids, ref['grid_thws'])} {ref['grid_thws'][0].tolist()}, "
              + (f"pixels max diff {diff.max().item():.3e}, over 1e-6: {(diff > 1e-6).sum().item()}/{diff.numel()}" if diff is not None
                 else f"pixel shapes differ {tuple(titan_pixels.shape)} vs {tuple(ref_pixels.shape)}"))


if __name__ == "__main__":
    if "--skip-a" not in sys.argv:
        part_a()
    part_b([p for p in sys.argv[1:] if p != "--skip-a"])
