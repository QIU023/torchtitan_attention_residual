"""Controlled evidence that the vision tower reaches the policy's logits.

One process, one set of weights, one token stream; the only variable is whether
``pixel_values`` is passed to the forward. Causality then pins the answer:

- positions before the first media pad must be bitwise identical (nothing upstream
  of the image can depend on it),
- the media pads and every position after them must differ (their embeddings come
  from the tower in one run and from the token table in the other).

All identical means the images never reached the model. A difference before the
first pad means something is wrong with the packing, not with the tower.

    torchrun --nproc_per_node=1 k3_vision_causal_probe.py [--export <k3 export>]
"""

from __future__ import annotations

import argparse
import sys

import torch


def build_batch(export: str, device: torch.device):
    """One image prompt through the released processor: ids, pixel values, grid."""
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(export, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(export, trust_remote_code=True)
    image = Image.new("RGB", (84, 56), (10, 200, 30))
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "What colour is this?"}],
        }
    ]
    sys.path.insert(0, "/tmp/wt_verl_0915")
    from verl.utils.tokenizer.tokenizer import build_multimodal_processor_inputs

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = build_multimodal_processor_inputs(processor, text=text, images=[image])
    ids = out["input_ids"][0].to(device)
    pad_id = int(processor.tokenizer.convert_tokens_to_ids("<|media_pad|>"))
    pixel_values = out["pixel_values"].to(device)
    # The processor emits (N, 3, 14, 14) and the patch embedding takes (N, 588); the
    # engine flattens it in _model_multimodal_kwargs and so does this probe.
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.flatten(1)
    return ids, pixel_values, out["grid_thws"].to(device), pad_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="/root/models/kimi-k3-debug-nt-rel")
    args = ap.parse_args()

    from torchtitan.models.kimi_k3 import model_registry

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    ids, pixel_values, grid_thw, pad_id = build_batch(args.export, device)

    spec = model_registry("debugmodel", seq_len=4096)
    # The tree's own order: build on meta, move, then let the model seed its
    # weights and buffers. Building straight on the device leaves them unset.
    with torch.device("meta"):
        model = spec.model.build()
    model.to_empty(device=device)
    init_weights = getattr(model, "init_weights", None)
    if init_weights is not None:
        init_weights(buffer_device=device)
    # The weights stay in fp32 and the forward runs under autocast, which is what the
    # trainer and the veRL engine do. Casting the module itself would also cast the
    # vision tower's rope cache, and torch.polar refuses bf16.
    model.eval()

    tokens = ids.to(torch.int64)
    positions = torch.arange(tokens.numel(), device=device, dtype=torch.int64)
    masks = model.get_attention_masks(positions=positions)
    pads = (tokens == pad_id).nonzero().flatten()
    if pads.numel() == 0:
        raise SystemExit("the prompt carries no media pad; check the processor call")
    first_pad = int(pads[0])
    print(f"[probe] tokens {tokens.numel()}, media pads {pads.numel()} at {pads.tolist()}", flush=True)

    kwargs = dict(positions=positions, attention_masks=masks)
    def logits(**extra) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(tokens, **kwargs, **extra)
        # A stage that does not own the head returns its hidden states and stack.
        if isinstance(out, tuple):
            raise SystemExit("the model returned stage outputs; run this without pipeline parallelism")
        return out.float()

    with_images = logits(
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        special_tokens={"image_id": pad_id},
    )
    without_images = logits()

    before = slice(0, first_pad)
    after = slice(first_pad, tokens.numel())
    same_before = torch.equal(with_images[before], without_images[before])
    d_before = (with_images[before] - without_images[before]).abs().max().item() if first_pad else 0.0
    d_after = (with_images[after] - without_images[after]).abs().max().item()
    print(f"[probe] positions before the first pad ({first_pad}): bitwise equal {same_before}, max diff {d_before:.3e}")
    print(f"[probe] the pads and everything after ({tokens.numel() - first_pad}): max diff {d_after:.3e}")
    verdict = "PASS" if same_before and d_after > 0 else "FAIL"
    print(f"[probe] {verdict}: the tower's features change the logits from the first media pad onward and nothing before it")
    if verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
