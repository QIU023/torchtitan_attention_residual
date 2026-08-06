"""Load a K3-family checkpoint in the OFFICIAL vLLM K3 implementation and generate.

The rollout half of the veRL GRPO gap. veRL's RL path has no naive rollout --
`_ROLLOUT_REGISTRY` holds only vllm/sglang/trtllm async servers -- so a K3 GRPO
run needs an inference engine that (a) has the model and (b) accepts the weight
names our trainer exports. This script checks both against the official
implementation rather than against the predecessor `kimi_linear.py`, which is a
different model and would pass for the wrong reason.

Official implementation: vllm-project/vllm, package `vllm/models/kimi_k3/`
(nvidia/ and amd/ variants), registering `KimiK3ForConditionalGeneration` and
`KimiK3MTPModel`. It also re-points `KimiLinearForCausalLM` at that package, so a
text-only K3-family checkpoint loads through the same code.

Env: needs a venv where `vllm` IS that build, not a pip release. As of vLLM
0.26.0 the released wheel has no K3 at all -- only `kimi_linear.py`,
`kimi_k25.py`, `kimi_vl.py`. Pointing this at a pip vllm silently tests the
predecessor.

    VERL_VLLM_VERSION=0.27.0 /venv/vllm_k3/bin/python vllm_k3_rollout_smoke.py <model_dir>

`VERL_VLLM_VERSION` matters only when verl is imported in the same process: a
source build reports a setuptools-scm placeholder ("0.1.dev1+g<sha>") that verl's
>= 0.7.0 gate rejects.
"""

import os
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/workspace/k3mini_hf_vllm"


def main() -> int:
    if os.environ.get("K3_SITU_SHIM") == "1":
        import situ_op_shim

        situ_op_shim.install()
        print("situ_and_mul: Python shim installed (contract check, not fidelity)")

    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.config import get_config

    # Multimodal needs remote code: the image processor is the release's
    # KimiK3VisionProcessor, not a transformers built-in.
    trust = os.environ.get("K3_TRUST_REMOTE_CODE") == "1"
    cfg = get_config(MODEL, trust_remote_code=trust)
    print(f"config class : {type(cfg).__name__}")
    print(f"architectures: {getattr(cfg, 'architectures', None)}")
    # KimiK3Config (multimodal) delegates hidden_size/vocab_size to text_config
    # and does not expose num_hidden_layers at the top level.
    inner = getattr(cfg, "text_config", cfg)
    print(f"layers/hidden: {inner.num_hidden_layers} / {cfg.hidden_size}")

    # Small numbers on purpose: this asks "does the official class load our
    # names and step the KDA/MLA decode path", not "how fast is it".
    llm = LLM(
        model=MODEL,
        trust_remote_code=trust,
        gpu_memory_utilization=0.30,
        max_model_len=256,
        enforce_eager=True,
        dtype="bfloat16",
    )
    which = type(llm.llm_engine.model_config.hf_config).__name__
    print(f"engine built, hf_config={which}")

    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    for o in out:
        print(f"prompt_tokens={len(o.prompt_token_ids)} "
              f"output_ids={list(o.outputs[0].token_ids)}")
    # A random-init fixture generates noise; the pass condition is that the
    # official class accepted every weight and the decode path ran.
    print("ROLLOUT SMOKE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
