#!/usr/bin/env python3
"""MiniPLM-style offline corpus scoring.

For each c4-en chunk, compute the per-token mean log-prob under
teacher (Llama-3.1-8B) and reference (Llama-3.2-1B), and the
"difference score":

    score = mean_log_p_teacher(chunk) - mean_log_p_reference(chunk)

High scores ≈ chunks where the larger teacher knows much more than
the small reference. Per MiniPLM, these chunks carry the densest
distillation signal, and continuing pretraining on a high-score
subset transfers that signal to the student WITHOUT running the
teacher in the train loop.

Usage (run 4 of these in parallel, one per GPU):

    CUDA_VISIBLE_DEVICES=0 python score_corpus.py \
        --teacher /root/hf_cache/models--NousResearch--Meta-Llama-3.1-8B \
        --reference NousResearch/Llama-3.2-1B \
        --shard 0 --num-shards 4 \
        --num-chunks 125000 --batch-size 8 --seq-len 2048 \
        --out scored_0.jsonl

Output: jsonl with one row per chunk:
    {"input_ids": [...], "teacher_logp": float, "reference_logp": float, "score": float}

Memory budget per GPU (single-rank, no FSDP):
    teacher 8B bf16    16 GB
    reference 1B bf16   2 GB
    activations         ~5 GB (B=8 T=2048)
    total              ~23 GB / 31 GB on 5090
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, help="HF id or local path")
    p.add_argument("--reference", required=True, help="HF id or local path")
    p.add_argument("--cache-dir", default="/root/hf_cache")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--num-chunks", type=int, default=125_000,
                   help="Chunks to score in this shard")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--out", required=True, help="Output jsonl path")
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


@torch.no_grad()
def chunk_loglik(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Mean per-token log-likelihood of input_ids under model.

    Returns one scalar per row in batch (shape [B]).

    Memory note: keep math in bf16 throughout. Upcasting the
    (B, T-1, V=128256) logits tensor to fp32 doubles VRAM (~8 GB at
    B=8) and OOMs the 8B teacher + 1B reference combo on a 31 GB
    5090. F.cross_entropy is a fused kernel that doesn't materialize
    a full log_softmax tensor; we use it with reduction='none' and
    keep the per-token NLL in bf16, then reduce to fp32 only on the
    [B] output.
    """
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :]                    # (B, T-1, V) bf16
    targets = input_ids[:, 1:]                        # (B, T-1)
    # Fused CE in bf16 — avoids full fp32 vocab tensor materialization.
    ce = F.cross_entropy(
        logits.flatten(0, 1),                         # (B*(T-1), V) bf16
        targets.flatten(0, 1),                        # (B*(T-1),)
        reduction="none",
    )                                                  # (B*(T-1),) bf16
    nll = ce.view(input_ids.size(0), -1)              # (B, T-1) bf16
    return -nll.float().mean(dim=-1)                  # (B,) fp32 — log p


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    print(f"[score] shard={args.shard}/{args.num_shards}  num_chunks={args.num_chunks}  device={device}", flush=True)

    # --- Tokenizer (must match student/teacher: Llama-3.1 BPE) ---
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    eos = tok.eos_token_id or 0

    # --- Models ---
    print(f"[score] loading teacher {args.teacher}", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        cache_dir=args.cache_dir, trust_remote_code=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[score] loading reference {args.reference}", flush=True)
    reference = AutoModelForCausalLM.from_pretrained(
        args.reference, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        cache_dir=args.cache_dir, trust_remote_code=True,
    ).to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    print(f"[score] models ready, mem={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # --- c4 stream, sharded across ranks via ds.shard ---
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    # Stream sharding: each rank takes every num_shards-th example.
    # Disjoint coverage, no double-counting, no need for explicit skips.
    ds = ds.shard(num_shards=args.num_shards, index=args.shard)

    # --- Tokenize-and-pack into seq_len chunks ---
    buf: list[int] = []
    chunk_buf: list[list[int]] = []
    n_written = 0
    t0 = time.perf_counter()

    out_f = out_path.open("w")
    try:
        for sample in ds:
            ids = tok.encode(sample["text"], add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= args.seq_len:
                chunk_buf.append(buf[: args.seq_len])
                buf = buf[args.seq_len:]
                if len(chunk_buf) >= args.batch_size:
                    batch = torch.tensor(chunk_buf[:args.batch_size],
                                         dtype=torch.long, device=device)
                    chunk_buf = chunk_buf[args.batch_size:]
                    teacher_logp = chunk_loglik(teacher, batch)
                    reference_logp = chunk_loglik(reference, batch)
                    score = teacher_logp - reference_logp
                    teacher_cpu = teacher_logp.detach().cpu().tolist()
                    reference_cpu = reference_logp.detach().cpu().tolist()
                    score_cpu = score.detach().cpu().tolist()
                    ids_cpu = batch.cpu().tolist()
                    for i in range(len(ids_cpu)):
                        out_f.write(json.dumps({
                            "input_ids": ids_cpu[i],
                            "teacher_logp": teacher_cpu[i],
                            "reference_logp": reference_cpu[i],
                            "score": score_cpu[i],
                        }) + "\n")
                        n_written += 1
                        if n_written % args.log_every == 0:
                            dt = time.perf_counter() - t0
                            tps = n_written * args.seq_len / dt
                            eta = (args.num_chunks - n_written) * args.seq_len / max(tps, 1)
                            print(f"[score] shard={args.shard}  written={n_written}/{args.num_chunks}  "
                                  f"tps={tps:.0f}  eta={eta/60:.1f}min  "
                                  f"recent_score_mean={sum(score_cpu)/len(score_cpu):.3f}",
                                  flush=True)
                        if n_written >= args.num_chunks:
                            return
    finally:
        out_f.close()
        print(f"[score] done. wrote {n_written} chunks to {out_path}", flush=True)


if __name__ == "__main__":
    main()
