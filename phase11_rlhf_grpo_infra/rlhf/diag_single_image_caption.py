#!/usr/bin/env python3
"""Decisive image-blind diagnostic: boot the SGLang AttnRes-VL engine on the
converted ckpt, caption ONE known image, print the output. If grounded -> image
injection works (rollout-path bug elsewhere). If garbage/ungrounded -> the
converted LM/projector in the SGLang load path is the bug. Mirrors the GRPO
rollout (same env + flashinfer + cuda_graph off + MLA fp32 fallback).

NOTE: SGLang Engine spawns scheduler subprocesses (spawn), which re-import this
module — so ALL engine/generate work MUST live under `if __name__ == '__main__'`.
"""
import os, base64

os.environ.setdefault("ATTNRES_MLA_FP32_FALLBACK", "1")
os.environ.setdefault("SGLANG_DISABLE_SHM_MM", "1")
os.environ.setdefault("SGLANG_FP8_IGNORED_LAYERS",
                      "attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts")

HF = "/workspace/torchtitan_attention_residual/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"
IMG = "/workspace/DriveLM_VLM_Project/data/nuscenes/samples_drivelm/val/val_data/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151062512404.jpg"


def main():
    import sglang.srt.configs.kimi_attn_res_vl  # noqa: F401  register config/model
    from sglang.srt.entrypoints.engine import Engine

    print("[diag] booting Engine on", HF, flush=True)
    eng = Engine(
        model_path=HF,
        skip_tokenizer_init=False,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=0.6,
        attention_backend="flashinfer",
        decode_attention_backend="torch_native",  # avoid bf16 MLA NaN (-> all-'!' garbage)
        linear_attn_backend="triton",
        disable_cuda_graph=True,
        log_level="error",
        base_gpu_id=0,
    )
    print("[diag] engine up. generating...", flush=True)

    SYS = ("You are a helpful vision assistant. Describe the image in one short "
           "sentence.")
    prompt = f"{SYS}\n\nUser: Describe the image briefly.\nAssistant:"
    sp = {"temperature": 0.0, "max_new_tokens": 40}

    with open(IMG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    img_durl = f"data:image/jpeg;base64,{b64}"

    import asyncio

    def _txt(o):
        o = o[0] if isinstance(o, list) else o
        return o.get("text") if isinstance(o, dict) else o

    async def _gen():
        # single-example path: prompt as STRING (is_single=True -> no batch expand)
        oi = await eng.async_generate(prompt, image_data=img_durl, sampling_params=sp)
        print("\n===== WITH IMAGE =====", flush=True)
        print(repr(_txt(oi)), flush=True)
        ot = await eng.async_generate(prompt, sampling_params=sp)
        print("\n===== TEXT-ONLY (no image) =====", flush=True)
        print(repr(_txt(ot)), flush=True)

    asyncio.run(_gen())
    print("\n[diag] done", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
