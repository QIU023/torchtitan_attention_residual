"""Export the multimodal mini checkpoint to HF format.

The point is that config, weights and `architectures` finally agree: the
checkpoint carries a real vision tower (390 vision_tower.* keys), so declaring
KimiK3ForConditionalGeneration is honest rather than aspirational. The previous
/workspace/k3mini_hf claimed to be multimodal while containing zero vision
tensors, which is what vLLM's multimodal path tripped over.

Usage: python export_vl_to_hf.py --dcp <step-N dir> --out <hf dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file


def load_dcp(path: str) -> dict:
    """Materialize a DCP checkpoint into a plain state dict."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    md = FileSystemReader(path).read_metadata()
    # Model tensors only -- optimizer, dataloader and train_state are not part
    # of an HF export.
    keys = [
        k for k in md.state_dict_metadata
        if k.split(".")[0] in ("language_model", "vision_tower", "lm_head",
                               "model")
    ]
    sd = {k: torch.empty(md.state_dict_metadata[k].size,
                         dtype=md.state_dict_metadata[k].properties.dtype)
          for k in keys}
    dcp.load(sd, checkpoint_id=path)
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-config", default="/workspace/k3mini_hf")
    args = ap.parse_args()

    sd = load_dcp(args.dcp)
    n_vision = sum(1 for k in sd if k.startswith("vision_tower"))
    print(f"loaded {len(sd)} tensors, {n_vision} from the vision tower")
    if n_vision == 0:
        raise SystemExit(
            "refusing to export: no vision_tower tensors, so declaring "
            "KimiK3ForConditionalGeneration would repeat the inconsistency "
            "this export exists to fix"
        )

    os.makedirs(args.out, exist_ok=True)
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, os.path.join(args.out, "model.safetensors"))

    # config: text fields nested under text_config, vision under vision_config,
    # auto_map pointing at the MULTIMODAL class so transformers does not resolve
    # the text-only one and hand vLLM a flat config.
    src = json.load(open(os.path.join(args.src_config, "config.json")))
    wrapper = {"architectures", "auto_map", "model_type", "torch_dtype",
               "transformers_version", "pad_token_id"}
    text = {k: v for k, v in src.items() if k not in wrapper}
    cfg = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "auto_map": {"AutoConfig": "configuration_kimi_k3.KimiK3Config"},
        "text_config": text,
        "vision_config": {
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "qkv_hidden_size": 384,
            "intermediate_size": 1024,
            "text_hidden_size": text.get("hidden_size", 512),
        },
    }
    for k in ("torch_dtype", "transformers_version", "pad_token_id"):
        if k in src:
            cfg[k] = src[k]
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    for f in ("configuration_kimi_k3.py", "tokenizer.json",
              "tokenizer_config.json"):
        s = os.path.join(args.src_config, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(args.out, f))

    print(f"wrote {args.out}: {len(sd)} tensors, config with text_config + "
          f"vision_config")


if __name__ == "__main__":
    main()
