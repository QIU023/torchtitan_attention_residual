"""Sanity check: SGLang Engine on the LM-only converted ckpt."""
import sys
import time


def main():
    LM_CKPT = sys.argv[1] if len(sys.argv) > 1 else (
        "/root/torchtitan_attention_residual/phase11/hf/lm_base"
    )
    import sglang as sgl
    t0 = time.perf_counter()
    import os
    e = sgl.Engine(
        model_path=LM_CKPT,
        tp_size=1,
        dtype="bfloat16",
        attention_backend="flashinfer",
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        disable_radix_cache=True,
        disable_cuda_graph=bool(int(os.environ.get("DISABLE_CUDA_GRAPH", "0"))),
    )
    print(f"[lm-smoke] booted in {time.perf_counter()-t0:.1f}s")
    out = e.generate(
        prompt="Hello, my name is",
        sampling_params={"temperature": 0.0, "max_new_tokens": 20},
    )
    print(f"[lm-smoke] gen: {out.get('text', '')!r}")
    print("[lm-smoke] DONE")


if __name__ == "__main__":
    main()
