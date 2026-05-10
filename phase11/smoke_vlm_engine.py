"""End-to-end SGLang Engine smoke for the 447M VLM AttnRes model.

Brings up an Engine on TP=1, GPU 7 (avoiding the ones used by COCO
download / other work), feeds it an image+text prompt, and decodes a
short caption. Success here proves the full pipeline:

  config.json → arch lookup → KimiAttnResVLForConditionalGeneration
  load_weights → splits LM/projector/vision and binds correctly
  HF SigLIP → vision tower forward
  general_mm_embed_routine → vision feature splice into LM embeds
  KimiBlockAttnResForCausalLM → AttnRes decode
  Logits → tokenizer detokenization

The image-token padding path inside SGLang requires the chat template
to emit ``<image>`` placeholders that map to the configured
image_token_id (32000). We construct a minimal prompt that includes
the placeholder explicitly to avoid depending on a not-yet-shipped
chat template for our model.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path", type=Path,
        default=_WS / "phase11" / "hf/vlm_pretrain",
    )
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument(
        "--image-path", type=str,
        default=str(_WS / "phase5" / "tests" / "data" / "coco_dummy.jpg"),
        help="path to a test image; falls back to a random one in "
             "/workspace/.hf_home/LLaVA-Pretrain if the default is missing.",
    )
    args = p.parse_args()

    if not args.model_path.exists():
        print(f"ERROR: VLM ckpt not found at {args.model_path}")
        return 1

    image_path = Path(args.image_path)
    if not image_path.exists():
        # Fallback: any image from LLaVA-Pretrain
        for cand in (_WS.parent.parent / ".hf_home" / "LLaVA-Pretrain").glob(
            "00001/*.jpg"
        ):
            image_path = cand
            break
        else:
            for cand in Path("/workspace/.hf_home/LLaVA-Pretrain").glob(
                "00001/*.jpg"
            ):
                image_path = cand
                break
    if not image_path.exists():
        print("ERROR: no test image found")
        return 1
    print(f"[engine-smoke] image: {image_path}")

    print(f"[engine-smoke] booting SGLang Engine TP={args.tp_size}")
    t0 = time.perf_counter()
    import sglang as sgl

    engine = sgl.Engine(
        model_path=str(args.model_path),
        tp_size=args.tp_size,
        dtype="bfloat16",
        attention_backend="flashinfer",
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        # KDA's forward path requires the piecewise_context_manager's
        # forward_context to be active when calling
        # unified_linear_attention_with_output. CUDA graphs must be
        # enabled for that context to be set.
        disable_radix_cache=True,
    )
    boot_dt = time.perf_counter() - t0
    print(f"[engine-smoke] engine ready in {boot_dt:.1f}s")

    prompt = (
        "<image>\n"
        "Describe this image in one short sentence."
    )
    print(f"[engine-smoke] prompt: {prompt!r}")

    t0 = time.perf_counter()
    out = engine.generate(
        prompt=prompt,
        image_data=str(image_path),
        sampling_params={"temperature": 0.0, "max_new_tokens": 40},
    )
    gen_dt = time.perf_counter() - t0
    text = out.get("text", "")
    print(f"[engine-smoke] gen ({gen_dt:.2f}s): {text[:200]!r}")
    print("[engine-smoke] DONE ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
