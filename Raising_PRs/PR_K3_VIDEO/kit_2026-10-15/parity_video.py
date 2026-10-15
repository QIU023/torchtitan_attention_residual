"""Video-chunk forward parity: TorchTitan Kimi K3 vs the released HuggingFace code (local kit, never committed).

Builds on scripts/checkpoint_conversion/numerical_tests_kimi_k3.py: same pinned revision, reduced debug config,
strict weight transfer and bf16. The released processor has no video branch, so both sides take the same chunk
patches (TorchTitan's chunked video preprocessing) and the HF prompt spells each chunk as Kimi K2.5's processor
does: "<hh:mm:ss.fff><|media_begin|>video<|media_content|><|media_pad|><|media_end|>".
"""

import argparse
import importlib.abc
import importlib.machinery
import sys
import types
from typing import Any


class _Any:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Any()

    def __getattr__(self, name):
        return _Any()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any


class _FlaStub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """fla-core 0.5.2 fails to import on this Triton; the vision path never calls it."""

    def find_spec(self, name, path, target=None):
        if name == "fla" or name.startswith("fla."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []


sys.meta_path.insert(0, _FlaStub())

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

from scripts.checkpoint_conversion import numerical_tests_kimi_k3 as base
from torchtitan.components.data.multimodal.utils.image import resize_to_navit_patch_grid, vision_to_patches
from torchtitan.components.data.multimodal.utils.video import format_video_timestamp, process_video
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

_CHUNK = 4
_FPS = 8.0


def _frames(num_frames: int, size: int) -> torch.Tensor:
    t = torch.arange(num_frames).view(-1, 1, 1, 1)
    ramp = torch.linspace(0, 255, size * size * 3).reshape(1, size, size, 3)
    frames = ((ramp + 23 * t) % 256).to(torch.uint8)
    side = size // 4
    for i in range(num_frames):
        top, left = (i * 7) % (size - side), (i * 11) % (size - side)
        frames[i, top : top + side, left : left + side] = torch.tensor([255, 40 * (i % 6), 30], dtype=torch.uint8)
    return frames


def _chunks(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    video = process_video(
        frames,
        patch_size=base._PATCH_SIZE,
        merge_size=base._MERGE_SIZE,
        min_pixels=1,
        max_pixels=1 << 30,
        resize_fn=resize_to_navit_patch_grid,
        max_patches=16384,
        max_patches_per_side=base._MAX_PATCHES_PER_SIDE,
    )
    pieces = [
        vision_to_patches(chunk, base._PATCH_SIZE, 1, base._MERGE_SIZE, patch_order="raster")
        for chunk in video.split(_CHUNK)
    ]
    return torch.cat([p for p, _ in pieces]), torch.stack([g for _, g in pieces])


def _expand(input_ids: torch.Tensor, counts: list[int]) -> torch.Tensor:
    ids = input_ids[0].tolist()
    out, k = [], 0
    for token in ids:
        if token == base._MEDIA_TOKEN_ID:
            out += [token] * counts[k]
            k += 1
        else:
            out.append(token)
    assert k == len(counts), (k, counts)
    return torch.tensor(out)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    hf_model_path = snapshot_download(
        repo_id=base._HF_REPO_ID, revision=base._HF_REVISION, allow_patterns=["*.json", "*.py", "tiktoken.model"]
    )
    device, dtype = torch.device("cuda"), base._DTYPE
    tt_config = base._use_released_vocab(
        build_model_config("debugmodel", parallelism=ParallelismConfig(), attn_backend="flex"), hf_model_path
    )
    torch.manual_seed(args.seed)
    tt_model = base._build_tt_model(tt_config, dtype)
    hf_state_dict = KimiK3StateDictAdapter(tt_config, hf_assets_path=None).to_hf(tt_model.state_dict())

    patches, grids = _chunks(_frames(args.num_frames, args.size))
    counts = [int((g[1] // base._MERGE_SIZE) * (g[2] // base._MERGE_SIZE)) for g in grids]
    prompt = "".join(
        f"{format_video_timestamp(i * _CHUNK / _FPS)}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
        for i in range(len(counts))
    ) + "\nWhat happens in this video?"
    processor: Any = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True, local_files_only=True)
    input_ids = processor.tokenizer(prompt, return_tensors="pt")["input_ids"]
    assert int((input_ids == base._MEDIA_TOKEN_ID).sum()) == len(counts), input_ids
    print(f"frames={args.num_frames} size={args.size} grids={grids.tolist()} tokens/chunk={counts}")

    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True, local_files_only=True)
    base._reduce_hf_config(hf_config, tt_config, hf_model_path)
    hf_config.text_config._attn_implementation = base._HF_ATTN_BACKEND
    hf_config.vision_config._attn_implementation = base._HF_ATTN_BACKEND
    hf = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    assert {b.attn_implementation for b in hf.vision_tower.encoder.blocks} == {base._HF_ATTN_BACKEND}
    hf.to(dtype=dtype)
    missing, unexpected = hf.load_state_dict(hf_state_dict, strict=False)
    vision = [k for k in list(missing) + list(unexpected) if k.startswith(("vision_tower.", "mm_projector."))]
    assert not vision, vision
    loaded = [k for k in hf_state_dict if k.startswith(("vision_tower.", "mm_projector."))]
    print(f"HF vision/projector tensors loaded: {len(loaded)}")
    hf = hf.eval().to(device)
    feats: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        out = torch.cat(list(out)) if isinstance(out, (list, tuple)) else out
        feats["hf"] = out.detach().float().cpu()

    hf.mm_projector.register_forward_hook(hook)
    hf_pixels = patches.reshape(-1, 3, base._PATCH_SIZE, base._PATCH_SIZE).to(device, dtype)
    hf.mm_projector(hf._extract_image_features(hf_pixels, grids.to(device)))
    del hf
    torch.cuda.empty_cache()

    model = tt_model.to(device)
    pixel_values, grid_thw = patches.to(device, dtype), grids.to(device)
    tt_feats = model.vision_encoder(pixel_values, grid_thw=grid_thw).float().cpu()
    hf_feats = feats["hf"].reshape(tt_feats.shape)
    cos = F.cosine_similarity(hf_feats.flatten(), tt_feats.flatten(), dim=0).item()
    print(f"vision features: shape={tuple(tt_feats.shape)} cos={cos:.6f} max_diff={(hf_feats - tt_feats).abs().max().item():.3e}")
    per_chunk = [int(c) for c in counts]
    for i, (hf_chunk, tt_chunk) in enumerate(zip(hf_feats.split(per_chunk), tt_feats.split(per_chunk))):
        c = F.cosine_similarity(hf_chunk.flatten(), tt_chunk.flatten(), dim=0).item()
        print(f"  chunk {i}: t={int(grids[i][0])} cos={c:.6f} max_diff={(hf_chunk - tt_chunk).abs().max().item():.3e}")
    tokens = _expand(input_ids, counts).to(device)
    positions = torch.arange(tokens.shape[0], dtype=torch.int32, device=device)
    embeds = model._prepare_multimodal_embeds(
        tokens, pixel_values=pixel_values, grid_thw=grid_thw, special_tokens={"image_id": base._MEDIA_TOKEN_ID}
    )
    rows = (tokens == base._MEDIA_TOKEN_ID).nonzero().flatten()
    scattered = embeds[rows].float().cpu()
    print(f"scatter: {rows.numel()} placeholder rows, max_diff vs HF features {(scattered - hf_feats).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
