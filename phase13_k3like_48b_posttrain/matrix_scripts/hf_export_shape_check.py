"""Check an exported K3 HF config against the tensors it ships with.

The debug exports start from a copy of the reference config, so a field the
downscaled model never exercises keeps the RELEASED value. Nothing catches
that until an engine reads the field: vLLM allocated 896 experts (OOM) and
sized the latent projection at 3584 (load-time shape assert), while the same
config had served dense and text-only exports for weeks -- those models have
no routed experts, so neither field was ever read.

Usage: hf_export_shape_check.py <model_dir> [<model_dir> ...]
Exit code is 1 if any directory disagrees with its own weights.
"""

import json
import sys
from pathlib import Path

from safetensors import safe_open

LAYERS = "language_model.model.layers"


def _shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


def _first(shapes: dict[str, tuple[int, ...]], suffix: str) -> tuple[str, tuple[int, ...]] | None:
    """The first layer carrying ``suffix``; absent means the model has no such module."""
    for key in sorted(shapes):
        if key.startswith(LAYERS) and key.endswith(suffix):
            return key, shapes[key]
    return None


def check(model_dir: Path) -> list[str]:
    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("text_config", config)
    shapes = _shapes(model_dir)
    if not shapes:
        return [f"{model_dir}: no safetensors to check against"]

    embed = shapes.get("language_model.model.embed_tokens.weight")
    layer_ids = {
        int(k[len(LAYERS) + 1 :].split(".")[0]) for k in shapes if k.startswith(LAYERS + ".")
    }

    # (config field, value from the weights, what reads it)
    expected: list[tuple[str, int | None, str]] = []
    if embed is not None:
        expected += [("vocab_size", embed[0], "embedding"), ("hidden_size", embed[1], "embedding")]
    if layer_ids:
        expected.append(("num_hidden_layers", max(layer_ids) + 1, "decoder stack"))
    for field, suffix, axis, reader in (
        ("routed_expert_hidden_size", "block_sparse_moe.routed_expert_down_proj.weight", 0,
         "latent MoE down projection"),
        ("q_lora_rank", "self_attn.q_a_proj.weight", 0, "MLA query low rank"),
        ("kv_lora_rank", "self_attn.kv_a_proj_with_mqa.weight", None, "MLA kv low rank"),
        ("intermediate_size", "mlp.gate_proj.weight", 0, "dense FFN"),
        ("moe_intermediate_size", "block_sparse_moe.experts.0.w1.weight", 0, "routed expert"),
    ):
        found = _first(shapes, suffix)
        if found is None:
            continue
        _, shape = found
        if field == "kv_lora_rank":
            # The checkpoint fuses kv_lora_rank with qk_rope_head_dim here.
            value = shape[0] - int(text.get("qk_rope_head_dim") or 0)
        else:
            value = shape[axis]
        expected.append((field, value, reader))

    experts = {
        int(k.split(".experts.")[1].split(".")[0])
        for k in shapes
        if ".experts." in k and k.split(".experts.")[1].split(".")[0].isdigit()
    }
    if experts:
        count = max(experts) + 1
        expected += [("num_experts", count, "vLLM FusedMoE"),
                     ("n_routed_experts", count, "HF modeling code")]

    problems = []
    for field, value, reader in expected:
        configured = text.get(field)
        if configured is None or value is None:
            continue
        if configured != value:
            problems.append(
                f"{model_dir.name}: {field} = {configured} but the weights say {value} ({reader})"
            )
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    problems = []
    for arg in argv:
        directory = Path(arg)
        if not (directory / "config.json").exists():
            print(f"SKIP {directory} (no config.json)")
            continue
        found = check(directory)
        print(f"{'BAD ' if found else 'OK  '} {directory}")
        problems += found
    for line in problems:
        print("  " + line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
