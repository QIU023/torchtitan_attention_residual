"""Qualitative eval for the SFT 3ep VLM ckpt.

Picks 10 random LLaVA-Pretrain images, runs the SGLang VLM engine on
them with both T=0 greedy and T=0.7 sampling, prints the generation +
checks for the EOS-trap pattern (output collapses to `!!!!` or starts
with non-letters indicating the model didn't attend to the image).

Exits with code 0 if **at least 6/10 samples produce coherent text
that starts with a letter and has < 30% '!' density**; nonzero
otherwise. This is the gate before launching the GRPO stage.

HISTORY (2026-05-12 GRPO v16 reward collapse):
  The 2026-05-11 run passed coherent=10/10 but downstream GRPO immediately
  saw reward_mean=-1.000 for 20+ steps. Root cause: this gate used `OR`
  (line 106) between T=0 and T=0.7, so any sample where greedy emitted
  coherent English but the T=0.7 sampler EOS-trapped to `word!!!!!!!` was
  still counted "coherent". GRPO uses T=0.7 sampling, so we must AND the
  two so a sample only passes if BOTH temperatures are non-garbage. Also
  switched to the exact GRPO prompt template (matches `run_grpo_llava_kimi.py`
  L193) so the gate exercises the same code path as GRPO rollouts.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

# Block AttnRes residual-stream magnitude grows unboundedly with depth
# (see phase11/VISION_INJECTION_BUG_RCA.md). On Blackwell (RTX 5090,
# SM 12.0) flashinfer_mla's bf16 internals overflow to NaN on the deep
# MLA layers — and with the VLM the merged image-token embeddings enter
# the LM ~40x larger than text embeds, so the first MLA layer (0-idx 3)
# already NaNs. Two complementary mitigations, both required:
#   - ATTNRES_MLA_FP32_FALLBACK=1  : fp32 eager MLA on the EXTEND/prefill
#                                    path (attn_res_overlay._mla_forward_fp32).
#   - decode_attention_backend=torch_native : eager SDPA for the decode
#                                    path (flashinfer_mla still NaNs at
#                                    decode; the fp32 fallback is
#                                    prefill-only). Set on the Engine
#                                    below.
# Set the env flag here so it is in place before the SGLang scheduler
# subprocess is spawned.
os.environ.setdefault("ATTNRES_MLA_FP32_FALLBACK", "1")

# Pre-import sglang overlays so AutoConfig.register('kimi_attn_res_vl', ...)
# fires before transformers.AutoConfig.from_pretrained reads the HF
# config.json model_type.
try:
    import sglang.srt.configs.kimi_attn_res_vl  # noqa: F401
    import sglang.srt.models.attn_res_vl_overlay  # noqa: F401
except ImportError:
    pass


def is_garbage(text: str) -> tuple[bool, str]:
    """Return (is_garbage, reason)."""
    stripped = text.strip()
    if not stripped:
        return True, "empty"
    if not re.match(r"^[A-Za-z\"]", stripped):
        return True, f"starts non-letter ({stripped[:5]!r})"
    bang_count = stripped.count("!")
    if bang_count / max(len(stripped), 1) > 0.3:
        return True, f"bang_density={bang_count}/{len(stripped)}"
    return False, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--images-dir", type=Path,
                   default=Path("/workspace/.hf_home/LLaVA-Pretrain"))
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--gate-threshold", type=float, default=0.6,
                   help="Min fraction of samples that must be 'coherent'")
    args = p.parse_args()

    # Sample images
    rng = random.Random(42)
    all_images = []
    for bucket in args.images_dir.glob("0000*/*.jpg"):
        all_images.append(bucket)
        if len(all_images) > 500:
            break
    if not all_images:
        print(f"[eval] no images under {args.images_dir}/00000*/")
        return 2
    selected = rng.sample(all_images, args.num_samples)
    print(f"[eval] sampled {len(selected)} images")

    from sglang.srt.entrypoints.engine import Engine
    print(f"[eval] booting SGLang Engine on {args.model_path}")
    t0 = time.perf_counter()
    e = Engine(
        model_path=str(args.model_path),
        tp_size=1,
        dtype="bfloat16",
        attention_backend="flashinfer",
        # flashinfer_mla bf16-NaNs on the deep AttnRes MLA layers at
        # decode on Blackwell; the ATTNRES_MLA_FP32_FALLBACK fp32 path is
        # prefill-only. Route decode-mode MLA through eager SDPA. See the
        # module-level comment + phase11/VISION_INJECTION_BUG_RCA.md.
        decode_attention_backend="torch_native",
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        disable_radix_cache=True,
        # torch_native attention has no init_cuda_graph_state, so CUDA
        # graph capture raises NotImplementedError. Disable it — this is
        # a small qualitative gate (a few dozen short generations), the
        # per-step latency hit is irrelevant here.
        disable_cuda_graph=True,
    )
    print(f"[eval] engine ready in {time.perf_counter()-t0:.1f}s")

    coherent = 0
    fail_reasons: list[str] = []
    for i, img in enumerate(selected):
        # Match the exact GRPO prompt template (run_grpo_llava_kimi.py L193)
        # so this gate exercises the same code path as GRPO rollouts.
        # GRPO system prompt: see LlavaCaptionTask._SYSTEM_PROMPT.
        prompt = (
            "You are a helpful vision assistant. Describe the image in one "
            "short\nsentence (5 to 30 words). Begin with a capital letter and "
            "end with a\nperiod.\n\n<image>\nUser: Describe the image briefly."
            "\nAssistant:"
        )
        # Greedy
        out = e.generate(
            prompt=prompt, image_data=str(img),
            sampling_params={"temperature": 0.0, "max_new_tokens": 50, "stop": []},
        )
        text_greedy = out.get("text", "").strip()
        garbage_g, reason_g = is_garbage(text_greedy)
        # Sampling (this matches GRPO's actual rollout temperature/top_p).
        out2 = e.generate(
            prompt=prompt, image_data=str(img),
            sampling_params={"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 50, "stop": []},
        )
        text_sample = out2.get("text", "").strip()
        garbage_s, reason_s = is_garbage(text_sample)

        # Sample is coherent ONLY IF BOTH temperatures produce non-garbage.
        # Previous OR-gate allowed broken-for-GRPO ckpts through because
        # greedy decoding can be coherent while T=0.7 EOS-traps to `!!!!`,
        # but GRPO uses T=0.7 so reward immediately collapses to -1.0.
        sample_coherent = (not garbage_g) and (not garbage_s)
        if sample_coherent:
            coherent += 1
            mark = "+"
        else:
            mark = "-"
            fail_reasons.append(f"img{i} greedy={reason_g} sample={reason_s}")
        print(f"[eval] {mark} img={img.name}")
        print(f"       T=0:   {text_greedy[:150]!r}")
        print(f"       T=0.7: {text_sample[:150]!r}")

    rate = coherent / args.num_samples
    print(f"\n[eval] coherent {coherent}/{args.num_samples} ({rate*100:.0f}%)")
    print(f"[eval] gate threshold: {args.gate_threshold*100:.0f}%")
    if fail_reasons:
        print("[eval] failures:")
        for r in fail_reasons:
            print(f"  {r}")
    e.shutdown()
    return 0 if rate >= args.gate_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
