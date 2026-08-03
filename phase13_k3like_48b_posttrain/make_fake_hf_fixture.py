#!/usr/bin/env python
"""Rebuild the /workspace/fake_hf/kimi_linear_194m veRL SFT fixture.

Recreates the box-local fixture the 07-19 smoke used (see
VERL_SFT_SMOKE_RUNBOOK.md "Checkpoint-dir requirements"): an HF-shaped
export of a RANDOM-INIT kimi_linear_194m flavor -- official aux files
(config structure, auto_map code, Kimi tokenizer) from the 48B snapshot,
dims swapped to the 194m flavor so the veRL engine auto-derives it, and
safetensors produced through KimiLinearStateDictAdapter.to_hf. Also
writes a tiny SFT parquet.

Run inside /venv/main with the fork on PYTHONPATH:
  python make_fake_hf_fixture.py --out /workspace/fake_hf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch


AUX_FILES = (
    "configuration_kimi.py",
    "modeling_kimi.py",
    "tiktoken.model",
    "tokenization_kimi.py",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "generation_config.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/fake_hf")
    ap.add_argument("--flavor-size", default="194m")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    from torchtitan.experiments.kimi_k3.model_configs import (
        build_kimi_linear_config,
    )
    from torchtitan.experiments.kimi_k3.model import KimiK3Model
    from torchtitan.experiments.kimi_k3.state_dict_adapter import (
        KimiLinearStateDictAdapter,
    )

    repo = "moonshotai/Kimi-Linear-48B-A3B-Base"
    model_dir = os.path.join(args.out, f"kimi_linear_{args.flavor_size}")
    os.makedirs(model_dir, exist_ok=True)

    # 1) Official aux files (code + tokenizer).
    for f in AUX_FILES:
        p = hf_hub_download(repo, f)
        shutil.copy(p, os.path.join(model_dir, f))

    # 2) config.json: official structure, 194m dims. The veRL engine
    # derives the flavor from (hidden_size, num_hidden_layers, vocab).
    cfg_path = hf_hub_download(repo, "config.json")
    with open(cfg_path) as fh:
        hf_cfg = json.load(fh)
    kc = build_kimi_linear_config(args.flavor_size)
    hf_cfg.update(
        {
            "hidden_size": kc.hidden_size,
            "num_hidden_layers": kc.num_hidden_layers,
            "num_attention_heads": kc.num_attention_heads,
            "num_key_value_heads": kc.num_key_value_heads,
            "kv_lora_rank": kc.kv_lora_rank,
            "qk_nope_head_dim": kc.qk_nope_head_dim,
            "qk_rope_head_dim": kc.qk_rope_head_dim,
            "v_head_dim": kc.v_head_dim,
            "moe_intermediate_size": kc.moe_intermediate_size,
            "intermediate_size": kc.intermediate_size,
            "num_experts": kc.num_experts,
            "num_experts_per_tok": kc.num_experts_per_token,
            "vocab_size": kc.vocab_size,
            "first_k_dense_replace": kc.first_k_dense_replace,
            "kda_layers": list(kc.kda_layers),
            "full_attn_layers": list(kc.full_attn_layers),
            "linear_attn_config": {
                "kda_layers": list(kc.kda_layers),
                "full_attn_layers": list(kc.full_attn_layers),
                "head_dim": kc.kda_head_dim,
                "num_heads": kc.kda_num_heads,
                "short_conv_kernel_size": kc.kda_short_conv_kernel_size,
            },
        }
    )
    with open(os.path.join(model_dir, "config.json"), "w") as fh:
        json.dump(hf_cfg, fh, indent=2)

    # 3) titan-side tokenizer.json placeholder (Trainer build requirement;
    # veRL's data path uses the Kimi tiktoken files instead).
    tt_tok = os.path.join(
        os.path.dirname(__file__), "..", "torchtitan", "tests", "assets",
        "tokenizer", "tokenizer.json",
    )
    shutil.copy(tt_tok, os.path.join(model_dir, "tokenizer.json"))

    # 4) Random-init flavor weights -> HF layout via to_hf.
    torch.manual_seed(42)
    with torch.device("cpu"):
        model = KimiK3Model(kc)
    model.init_weights()
    model = model.to(torch.bfloat16)
    from torchtitan.experiments.kimi_k3 import KimiK3Spec

    adapter = KimiLinearStateDictAdapter(
        KimiK3Spec(kimi_config=kc, num_blocks=None), hf_assets_path=None
    )
    hf_sd = adapter.to_hf(model.state_dict())
    from safetensors.torch import save_file

    shard = os.path.join(model_dir, "model-00001-of-00001.safetensors")
    # .clone() breaks tied-weight aliasing (lm_head/embed) safetensors rejects.
    save_file({k: v.contiguous().clone() for k, v in hf_sd.items()}, shard)
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in hf_sd.values())},
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in hf_sd},
    }
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)

    # 5) Tiny SFT parquet for the smoke.
    import pandas as pd

    rows = []
    for i in range(64):
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"What is {i} + {i}?"}],
                "response": f"{i} + {i} = {2 * i}.",
            }
        )
    pd.DataFrame(rows).to_parquet(os.path.join(args.out, "sft_tiny.parquet"))
    print(f"fixture ready at {model_dir} ({len(hf_sd)} tensors)")


if __name__ == "__main__":
    main()
