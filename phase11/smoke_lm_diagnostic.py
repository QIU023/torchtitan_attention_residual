"""Diagnostic: is the LM-only model truly garbage, or is greedy decode
hitting a degenerate mode?"""
import sys
import time


def main():
    LM_CKPT = sys.argv[1] if len(sys.argv) > 1 else (
        "/root/torchtitan_attention_residual/phase11/hf/lm_base"
    )
    import sglang as sgl
    e = sgl.Engine(
        model_path=LM_CKPT,
        tp_size=1,
        dtype="bfloat16",
        attention_backend="flashinfer",
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        disable_radix_cache=True,
    )
    print("[lm-diag] Engine ready")
    print()

    prompts = [
        "The quick brown fox",
        "Once upon a time, there was a",
        "import torch\n\ndef forward(x):",
        "Hello, my name is",
        "The capital of France is",
    ]
    for sp in [
        {"temperature": 0.0, "max_new_tokens": 30},
        {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 30},
        {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 30},
    ]:
        print(f"=== sampling: {sp} ===")
        for p in prompts:
            out = e.generate(prompt=p, sampling_params=sp)
            txt = (out.get("text") or "").replace("\n", " ")
            print(f"  prompt={p!r}")
            print(f"  gen   = {txt[:120]!r}")
        print()


if __name__ == "__main__":
    main()
