"""Export a text K3 flavor to an official-schema HF dir, straight from the model.

Why not reuse export_vl_to_hf.py: that script copies its text config from an
existing config.json and writes the model's own parameter names verbatim. Both
are wrong for loading in an inference engine -- the config can disagree with the
weights it is shipped beside (that is exactly how the stale fixture ended up
claiming a full-rank KDA gate while carrying the low-rank pair), and the engine
keys on the RELEASED names, not ours.

Here both halves come from one place: `titan_config_to_official` for config.json
and `titan_to_official` for every key. Neither can drift from the other because
neither is hand-maintained.

    python export_text_to_hf.py --flavor kimi_k3_k3mini_block_attn_res \
        --out /workspace/k3mini_text_hf [--vocab-size 256]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file


def _text_only(key: str) -> str:
    """Drop the multimodal wrapper prefix.

    hf_key_map targets the RELEASED layout, which is multimodal, so it emits
    ``language_model.model.*``. A text-only checkpoint declares
    KimiLinearForCausalLM, whose parameters are ``model.*`` -- vLLM's
    KimiK3ForConditionalGeneration is the one that adds ``language_model.``
    (its WeightsMapper rewrites ``language_model.layers.`` ->
    ``language_model.model.layers.``). Loading a text-only export with the
    multimodal prefix fails with "no module or parameter named
    'language_model'".
    """
    return key.removeprefix("language_model.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavor", default="kimi_k3_k3mini_block_attn_res")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument(
        "--tokenizer-from",
        default="/workspace/k3mini_hf",
        help="dir to copy tokenizer files from; the engine needs a tokenizer "
        "even when the weights are a fixture",
    )
    args = ap.parse_args()

    from torchtitan.models.kimi_k3 import model_registry
    from torchtitan.models.kimi_k3.hf_key_map import (
        UnmappedKey,
        titan_config_to_official,
        titan_to_official,
    )

    spec = model_registry(args.flavor)
    if args.vocab_size is not None:
        import dataclasses as dc

        spec.model.kimi_config = dc.replace(
            spec.model.kimi_config, vocab_size=args.vocab_size
        )
    kc = spec.model.kimi_config
    num_blocks = spec.model.num_blocks

    torch.manual_seed(0)
    model = spec.model.build().to(torch.bfloat16).cuda()
    model.init_weights(buffer_device="cuda")

    kda_layers = set(kc.kda_layers)
    out: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in model.state_dict().items():
        value = value.detach()
        # One stacked expert tensor on our side is num_experts official keys, so
        # it has to be sliced here rather than renamed. Everything else is 1:1.
        if key.endswith(("w1_EFD", "w2_EDF", "w3_EFD")):
            for e in range(value.shape[0]):
                official = titan_to_official(key, kda_layers=kda_layers, expert_idx=e)
                out[_text_only(official)] = value[e].contiguous().cpu()
            continue
        try:
            official = titan_to_official(key, kda_layers=kda_layers)
        except UnmappedKey:
            skipped.append(key)
            continue
        out[_text_only(official)] = value.contiguous().cpu()

    if skipped:
        raise SystemExit(
            f"{len(skipped)} parameter(s) have no official name, so this export "
            f"would be silently incomplete: {skipped[:8]}"
        )

    os.makedirs(args.out, exist_ok=True)
    save_file(out, os.path.join(args.out, "model.safetensors"))

    cfg = titan_config_to_official(kc, num_blocks=num_blocks)
    cfg["architectures"] = ["KimiLinearForCausalLM"]
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    for f in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(args.tokenizer_from, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))

    print(f"wrote {args.out}: {len(out)} tensors")
    print(f"  gate form: use_full_rank_gate="
          f"{cfg['linear_attn_config']['use_full_rank_gate']}, "
          f"checkpoint carries "
          f"{'g_proj' if any(k.endswith('g_proj.weight') for k in out) else 'g_a_proj/g_b_proj'}")


if __name__ == "__main__":
    main()
