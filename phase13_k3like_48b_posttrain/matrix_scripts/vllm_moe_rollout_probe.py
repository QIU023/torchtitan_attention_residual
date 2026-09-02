"""Does the newer vLLM serve the K3 debug model with its MoE stack restored?

The rollout side of GRPO was recorded as blocked on SITU kernel coverage. That
was measured against the vLLM in venv_verl (d6938b740), whose unquantized
triton MoE path predates SITU. This probe asks the question against the newer
build in isolation: no verl, no training loop, one generation.
"""

import os
import sys
import traceback

MODEL = "/root/models/kimi-k3-debug"


def main() -> int:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=256,
        gpu_memory_utilization=float(os.environ.get("PROBE_UTIL", "0.85")),
        tensor_parallel_size=int(os.environ.get("PROBE_TP", "1")),
        dtype="bfloat16",
        **({"quantization": os.environ["PROBE_QUANT"]} if os.environ.get("PROBE_QUANT") else {}),
    )
    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    print("GENERATED:", repr(out[0].outputs[0].text))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
