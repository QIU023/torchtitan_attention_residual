#!/usr/bin/env python
"""Convert a bf16 (gated_)lora DCP checkpoint to the packed-MXFP4 layout.

Streams tensor-by-tensor: reads each key individually from the source DCP
checkpoint (peak RAM ~= one tensor + the growing output dict of PACKED
tensors), quantizes every LoRA base weight (``*.base.weight`` with
in_features % 32 == 0) to MXTensor(fp4 e2m1 x2, block 32) split storage,
and writes a new DCP checkpoint whose keys match the
``lora_quantize_base='mxfp4'`` meta-built model:

    <prefix>.base.weight  ->  <prefix>.base_qdata  (uint8, [out, in/2])
                              <prefix>.base_scale  (e8m0 viewed uint8, [out, in/32])
    everything else       ->  copied through unchanged

Together with the meta packed-layout build (lora.py quantize_base_mxfp4
meta path) this is the quantize-then-shard path for QLoRA on meta-first
torchtitan: no rank ever materializes the full bf16 model. For 48B from
HF safetensors, run the existing hf->dcp conversion first, then this.

Usage:
  python stream_quantize_mxfp4_dcp.py --src <ckpt>/step-N --dst <out_dir>
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
from torchao.prototype.mx_formats.mx_tensor import MXTensor


def _read_keys(src: str) -> list[tuple[str, object]]:
    reader = FileSystemReader(src)
    meta = reader.read_metadata()
    return list(meta.state_dict_metadata.items())


def _load_one(src: str, key: str, tmeta) -> torch.Tensor:
    """Load a single tensor from the DCP checkpoint (streaming read)."""
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    if not isinstance(tmeta, TensorStorageMetadata):
        raise TypeError(f"{key}: non-tensor DCP entry ({type(tmeta).__name__})")
    buf = torch.empty(tmeta.size, dtype=tmeta.properties.dtype)
    flat = {key: buf}
    sd = unflatten_state_dict(flat, {key: key.split(".")})
    dcp.load(sd, storage_reader=FileSystemReader(src))
    return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source DCP dir (step-N)")
    ap.add_argument("--dst", required=True, help="output DCP dir")
    ap.add_argument(
        "--target-suffix", default=".base.weight",
        help="keys ending with this get quantized (LoRA-wrapped bases)",
    )
    args = ap.parse_args()

    # DCP save needs a (single-rank) process group.
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29399")
        dist.init_process_group("gloo", rank=0, world_size=1)

    out: dict[str, torch.Tensor] = {}
    n_packed = n_copied = 0
    for key, tmeta in _read_keys(args.src):
        t = _load_one(args.src, key, tmeta)
        if key.endswith(args.target_suffix) and t.shape[-1] % 32 == 0:
            prefix = key[: -len(args.target_suffix)]
            mx = MXTensor.to_mx(
                t.to(torch.bfloat16),
                elem_dtype=torch.float4_e2m1fn_x2,
                block_size=32,
            )
            out[f"{prefix}.base_qdata"] = mx.qdata.contiguous()
            out[f"{prefix}.base_scale"] = (
                mx.scale.view(torch.uint8).contiguous()
            )
            n_packed += 1
        else:
            out[key] = t
            n_copied += 1
        del t

    dcp.save(out, storage_writer=FileSystemWriter(args.dst))
    print(f"packed {n_packed} base weights, copied {n_copied} tensors -> {args.dst}")


if __name__ == "__main__":
    main()
