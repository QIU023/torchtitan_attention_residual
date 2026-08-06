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
        "--official-code",
        default="/workspace/k3_official_code",
        help="dir holding the release's preprocessor_config.json and "
        "kimi_k3_vision_processing.py, copied verbatim for --multimodal",
    )
    ap.add_argument(
        "--multimodal",
        action="store_true",
        help="emit the RELEASED layout: language_model.* plus vision_tower.*, "
        "nested text_config/vision_config, KimiK3ForConditionalGeneration. "
        "Without it the text-only layout is emitted for "
        "KimiLinearForCausalLM.",
    )
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
        titan_config_to_official_multimodal,
        titan_to_official,
    )

    # Two naming spaces: model_registry parses "<size>_<variant>" with variant in
    # baseline/block_attn_res/full_attn_res, while the debug and report-arch
    # flavors are config_registry FUNCTIONS whose names it cannot parse. Accept
    # either, so a flavor the trainer can run is a flavor this can export.
    try:
        spec = model_registry(args.flavor)
    except ValueError:
        from torchtitan.models.kimi_k3 import config_registry as _cr

        fn = getattr(_cr, args.flavor, None)
        if fn is None or not callable(fn):
            raise SystemExit(
                f"flavor {args.flavor!r} is neither a model_registry flavor nor a "
                "config_registry function"
            )
        spec = fn().model_spec
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
    # titan_to_official always emits the released (multimodal) spelling;
    # the text-only layout is that with the wrapper prefix removed.
    _spell = (lambda k: k) if args.multimodal else _text_only
    out: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in model.state_dict().items():
        value = value.detach()
        # A multimodal model's children are vision_tower and language_model, so
        # its text keys arrive as "language_model.layers.N...". titan_to_official
        # takes our INNER names, so strip the wrapper child prefix; vision_tower.*
        # it already handles.
        key = key.removeprefix("language_model.")
        # One stacked expert tensor on our side is num_experts official keys, so
        # it has to be sliced here rather than renamed. Everything else is 1:1.
        if key.endswith(("w1_EFD", "w2_EDF", "w3_EFD")):
            for e in range(value.shape[0]):
                official = titan_to_official(key, kda_layers=kda_layers, expert_idx=e)
                out[_spell(official)] = value[e].contiguous().cpu()
            continue
        try:
            official = titan_to_official(key, kda_layers=kda_layers)
        except UnmappedKey:
            skipped.append(key)
            continue
        if official.endswith(".A_log") and value.dim() == 1:
            # KDA A_log is [1, 1, H, 1] in the HF layout and [H] in ours. The
            # state-dict adapter reshapes on read and passes ours through on
            # write; this export maps names, so the shape has to be converted
            # here or the loader rejects it on size.
            value = value.view(1, 1, -1, 1)
        out[_spell(official)] = value.contiguous().cpu()

    if skipped:
        raise SystemExit(
            f"{len(skipped)} parameter(s) have no official name, so this export "
            f"would be silently incomplete: {skipped[:8]}"
        )

    os.makedirs(args.out, exist_ok=True)
    # Sharded name + index, not a bare model.safetensors: torchtitan's
    # StateDictAdapter regexes the shard number out of the index's weight_map, so
    # a single unindexed file fails with "model.safetensors.index.json not found".
    # vLLM accepts either.
    shard = "model-00001-of-00001.safetensors"
    save_file(out, os.path.join(args.out, shard))
    json.dump(
        {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in out.values())},
            "weight_map": {k: shard for k in out},
        },
        open(os.path.join(args.out, "model.safetensors.index.json"), "w"),
        indent=2,
    )

    if args.multimodal:
        cfg = titan_config_to_official_multimodal(
            kc,
            spec.model.vision_config,
            num_blocks=num_blocks,
            media_placeholder_token_id=spec.model.vision_token_id,
        )
    else:
        cfg = titan_config_to_official(kc, num_blocks=num_blocks)
        cfg["architectures"] = ["KimiLinearForCausalLM"]
    cfg["torch_dtype"] = "bfloat16"
    # transformers has no kimi_linear model type, and veRL's HFModelConfig goes
    # through AutoConfig -- without auto_map it fails with "Transformers does not
    # recognize this architecture". vLLM is unaffected either way: it resolves
    # model_type through its OWN registry and never consults auto_map.
    remote_cfg = "configuration_kimi_k3.py"
    have_remote = os.path.exists(os.path.join(args.tokenizer_from, remote_cfg))
    if have_remote and not args.multimodal:
        cfg["auto_map"] = {"AutoConfig": f"{remote_cfg[:-3]}.KimiLinearConfig"}
    elif args.multimodal:
        # Ship OUR minimal nested config module. The release has its own, which we
        # do not hold locally; the file we do have defines the TEXT config class,
        # and pointing AutoConfig at that for a nested kimi_k3 config hands
        # transformers the wrong schema. vLLM needs none of this -- it resolves
        # model_type through its own registry -- but anything going through
        # AutoConfig does, veRL's HFModelConfig included.
        mm_cfg = "configuration_kimi_k3_mm.py"
        shutil.copy(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), mm_cfg),
            os.path.join(args.out, mm_cfg),
        )
        cfg["auto_map"] = {"AutoConfig": f"{mm_cfg[:-3]}.KimiK3Config"}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    if args.multimodal:
        # vLLM's multimodal path resolves an image processor from the model dir
        # before it will even call the model multimodal-capable, so a multimodal
        # export without one fails in _create_processing_info long before any
        # weight is read.
        #
        # COPIED from the release, not generated. An earlier attempt wrote a
        # config naming "KimiK3ImageProcessor" -- a class that does not exist,
        # carried over from a hand-made fixture. The real artifact is structurally
        # different: no image_processor_type, an auto_map pointing at
        # KimiK3VisionProcessor, and the settings nested under media_proc_cfg.
        # Inventing this file is how a fixture ends up describing preprocessing
        # the tower was never built for.
        for f in ("preprocessor_config.json", "kimi_k3_vision_processing.py"):
            src = os.path.join(args.official_code, f)
            if not os.path.exists(src):
                raise SystemExit(
                    f"multimodal export needs {f} from the release; looked in "
                    f"{args.official_code}. Pass --official-code."
                )
            shutil.copy(src, os.path.join(args.out, f))

    for f in ("tokenizer.json", "tokenizer_config.json", remote_cfg):
        src = os.path.join(args.tokenizer_from, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))

    text_cfg = cfg["text_config"] if args.multimodal else cfg
    n_vision = sum(1 for k in out if k.startswith("vision_tower"))
    print(f"wrote {args.out}: {len(out)} tensors, {n_vision} vision")
    print(f"  arch: {cfg['architectures'][0]}")
    print(f"  gate form: use_full_rank_gate="
          f"{text_cfg['linear_attn_config']['use_full_rank_gate']}, "
          f"checkpoint carries "
          f"{'g_proj' if any(k.endswith('g_proj.weight') for k in out) else 'g_a_proj/g_b_proj'}")


if __name__ == "__main__":
    main()
