"""Inference dtype/quantization micro-bench for the Kimi-Linear AttnRes VLM.

Boots one SGLang Engine with the requested ``--dtype`` /
``--quantization`` / ``--decode-attention-backend``, runs N captioning
generations on real LLaVA-Pretrain images, and reports:

* engine boot time
* per-prompt prefill+decode wall time
* total generated tokens
* tokens/sec (throughput)
* coherent X/N (basic correctness gate; reuses the same is_garbage check
  as eval_sft_3ep_qualitative.py)

Run one config per process invocation — Engine instances don't compose
cleanly on the same GPUs.

Usage:

    python phase11_rlhf_grpo_infra/bench_inference_dtype.py \\
        --model-path phase5_vlm_multimodal_sft/runs/mm_sft_447m_full/hf_step3100 \\
        --dtype bfloat16 \\
        --decode-attention-backend torch_native \\
        --num-samples 8 \\
        --max-new-tokens 64
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

# Both env knobs that our SFT eval already sets — keep matrix apples-to-apples
# unless explicitly overridden.
os.environ.setdefault("ATTNRES_MLA_FP32_FALLBACK", "1")
# Exclude AttnRes pseudo-query projections (Linear(D, 1)) from fp8 quant.
# attn_res.py:128 docstring: "filter via filter_fqns to keep AttnRes
# pseudo-queries in high precision — the zero-init carrier story relies
# on small deltas accumulating, which rowwise FP8 quantization noise
# would destroy". Empirically, on RTX 5090 the fp8-quantized 1×D weights
# also crash phase-1 einsum's cuBLAS strided batched bf16 GEMM with
# CUBLAS_STATUS_EXECUTION_FAILED. Comma-separated; matches by dotted-
# component substring (Fp8Config + utils._module_path_match).
os.environ.setdefault(
    "SGLANG_FP8_IGNORED_LAYERS",
    # AttnRes pseudo-queries (see above) +
    # mlp.experts: skip fp8 path for the MoE — even with
    # UPSTREAM_PR_LIST #8's shmem-shrink config, the fp8 fused-MoE
    # Triton kernel hits "illegal memory access" on Blackwell (RTX 5090).
    # Falls back to UnquantizedFusedMoEMethod (bf16 MoE), which preserves
    # fp8 weight quant on the dense Linear layers (q/k/v/o, mlp gate/up/down)
    # while keeping MoE numerically safe.
    "attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts",
)

# Pre-import sglang overlays so AutoConfig.register('kimi_attn_res_vl', ...)
# fires before transformers reads the HF config.
try:
    import sglang.srt.configs.kimi_attn_res_vl  # noqa: F401
    import sglang.srt.models.attn_res_vl_overlay  # noqa: F401
    import sglang.srt.models.attn_res_overlay  # noqa: F401
except ImportError:
    pass


_NON_LETTER_PREFIX = re.compile(r"^[^A-Za-z]{4,}")


def is_garbage(text: str) -> tuple[bool, str]:
    if not text:
        return True, "empty"
    m = _NON_LETTER_PREFIX.match(text)
    if m:
        return True, f"starts non-letter ({text[:5]!r})"
    return False, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument(
        "--images-dir", type=Path,
        default=Path("/workspace/.hf_home/LLaVA-Pretrain"),
    )
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--quantization", default="",
        help="SGLang quantization arg: '', 'fp8', 'fp8_e4m3', 'awq', etc.",
    )
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument(
        "--decode-attention-backend", default="torch_native",
        help="Set to '' to use the same backend as attention-backend "
             "(i.e. fused flashinfer for decode too — known to NaN on "
             "undertrained AttnRes ckpts).",
    )
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="0.0 = greedy. Bench timing is greedy by default for repeatability.",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    all_images: list[Path] = []
    for bucket in args.images_dir.glob("0000*/*.jpg"):
        all_images.append(bucket)
        if len(all_images) > 500:
            break
    if not all_images:
        print(f"[bench] no images under {args.images_dir}/00000*/")
        return 2
    selected = rng.sample(all_images, args.num_samples)

    # --- Engine boot ---
    from sglang.srt.entrypoints.engine import Engine

    engine_kwargs = dict(
        model_path=str(args.model_path),
        tp_size=1,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        disable_radix_cache=True,
    )
    decode_backend = args.decode_attention_backend or args.attention_backend
    engine_kwargs["decode_attention_backend"] = decode_backend
    if decode_backend == "torch_native" or args.attention_backend == "torch_native":
        # torch_native has no init_cuda_graph_state.
        engine_kwargs["disable_cuda_graph"] = True
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization

    print(f"[bench] config: dtype={args.dtype} attn={args.attention_backend} "
          f"decode={decode_backend} quant={args.quantization or 'none'}")
    print(f"[bench] booting Engine on {args.model_path}")
    t0 = time.perf_counter()
    e = Engine(**engine_kwargs)
    boot_s = time.perf_counter() - t0
    print(f"[bench] engine ready in {boot_s:.1f}s")

    # --- Warm-up (1 generation, not measured) ---
    warmup_prompt = (
        "You are a helpful vision assistant. Describe the image in one "
        "short\nsentence (5 to 30 words). Begin with a capital letter and "
        "end with a\nperiod.\n\n<image>\nUser: Describe the image briefly."
        "\nAssistant:"
    )
    print("[bench] warmup generation...")
    _ = e.generate(
        prompt=warmup_prompt, image_data=str(selected[0]),
        sampling_params={"temperature": 0.0, "max_new_tokens": 16},
    )

    # --- Timed loop ---
    per_prompt_s: list[float] = []
    per_prompt_tokens: list[int] = []
    coherent = 0
    print(f"[bench] running {args.num_samples} timed generations "
          f"(max_new_tokens={args.max_new_tokens}, T={args.temperature})")
    for i, img in enumerate(selected):
        t1 = time.perf_counter()
        out = e.generate(
            prompt=warmup_prompt, image_data=str(img),
            sampling_params={
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "stop": [],
            },
        )
        dt = time.perf_counter() - t1
        text = out.get("text", "").strip()
        meta = out.get("meta_info", {}) or {}
        n_out = (
            meta.get("completion_tokens")
            or meta.get("output_tokens")
            or len(text.split())
        )
        per_prompt_s.append(dt)
        per_prompt_tokens.append(int(n_out))
        is_g, reason = is_garbage(text)
        if not is_g:
            coherent += 1
        mark = "+" if not is_g else "-"
        print(f"[bench] {mark} img{i:02d} dt={dt:.2f}s tok={n_out} "
              f"text={text[:80]!r}{'' if not is_g else f' [GARBAGE:{reason}]'}")

    total_s = sum(per_prompt_s)
    total_tokens = sum(per_prompt_tokens)
    avg_s = total_s / max(len(per_prompt_s), 1)
    tps = total_tokens / total_s if total_s > 0 else 0.0

    print()
    print("=" * 60)
    print(f"[bench] BOOT_S        = {boot_s:.1f}")
    print(f"[bench] TOTAL_S       = {total_s:.2f}  ({args.num_samples} prompts)")
    print(f"[bench] AVG_S/PROMPT  = {avg_s:.2f}")
    print(f"[bench] TOTAL_TOKENS  = {total_tokens}")
    print(f"[bench] TOKENS_PER_S  = {tps:.1f}")
    print(f"[bench] COHERENT      = {coherent}/{args.num_samples}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
