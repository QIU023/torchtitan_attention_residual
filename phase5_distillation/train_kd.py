# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KD training loop: 436M student × Kimi-Linear-48B-A3B-Base teacher.

Lives outside the torchtitan submodule on purpose — distillation
composes torchtitan's pieces (ModelSpec, parallelize, optimizer,
checkpointer, dataloader) but doesn't extend any of them.

Distributed assumptions:
* Single node, NGPU GPUs, 1-D mesh: dp_shard = NGPU.
* Both student and teacher FSDP2-shard across the same mesh.
* Each rank gets ``local_batch_size`` examples per microbatch; teacher
  forward runs locally on that shard.
* Backward and optimizer step run on student only; teacher params are
  frozen, FSDP just all-gathers them for forward.

Run via ``launch_kd.sh`` (torchrun + sensible defaults).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint import load as dcp_load, save as dcp_save
from torch.distributed.device_mesh import init_device_mesh

# Repo paths.
WORKSPACE = Path(__file__).resolve().parent.parent
TORCHTITAN_PATH = WORKSPACE / "torchtitan"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(TORCHTITAN_PATH))

from phase5_distillation.kd_loss import KDConfig, kd_loss  # noqa: E402
from phase5_distillation.teacher_runner import (  # noqa: E402
    DEFAULT_TEACHER,
    TeacherRunner,
)


# -----------------------------------------------------------------
# CLI args
# -----------------------------------------------------------------


@dataclass
class Args:
    student_config: str
    student_ckpt: str
    teacher: str
    output_dir: str
    steps: int
    local_bs: int
    global_bs: int
    seq_len: int
    lr: float
    kd_alpha: float
    kd_temperature: float
    log_freq: int
    save_freq: int
    teacher_cache_dir: str | None
    seed: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-config", required=True,
                   help="kimi_linear flavor name, e.g. "
                        "kimi_linear_436m_block_attn_res_n4")
    p.add_argument("--student-ckpt", required=True,
                   help="path to torchtitan student ckpt directory "
                        "(e.g. .../checkpoint/step-12500)")
    p.add_argument("--teacher", default=DEFAULT_TEACHER,
                   help="HF repo id or local path of the teacher")
    p.add_argument("--output-dir", required=True,
                   help="dir for KD run logs + ckpts")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--local-bs", type=int, default=2)
    p.add_argument("--global-bs", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="constant LR (distillation phase, post-cosine-decay)")
    p.add_argument("--kd-alpha", type=float, default=0.3)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--log-freq", type=int, default=10)
    p.add_argument("--save-freq", type=int, default=500)
    p.add_argument("--teacher-cache-dir", default=None,
                   help="optional HF cache dir override for the teacher")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Args(**vars(a))


# -----------------------------------------------------------------
# Distributed setup
# -----------------------------------------------------------------


def setup_distributed():
    """Initialize NCCL process group + return (rank, world, device, mesh)."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    return rank, world, device, mesh


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str):
    if is_rank0():
        print(f"[KD {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------
# Student build
# -----------------------------------------------------------------


def build_student(student_config: str, mesh, device: torch.device, seq_len: int):
    """Build + parallelize the student model from a torchtitan flavor."""
    from torchtitan.experiments.kimi_linear import model_registry
    from torchtitan.experiments.kimi_linear.parallelize import (
        parallelize_kimi_linear,
    )
    from torchtitan.config import (
        ParallelismConfig, TrainingConfig, CompileConfig,
        ActivationCheckpointConfig,
    )
    from torchtitan.distributed import ParallelDims

    spec = model_registry(student_config)

    # Build on meta, then move to device via FSDP wrap.
    with torch.device("meta"):
        student = spec.model.build()

    parallel_dims = ParallelDims(
        dp_shard=mesh.size(),
        dp_replicate=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=mesh.size(),
    )

    job_config_parallelism = ParallelismConfig(
        data_parallel_shard_degree=mesh.size(),
        data_parallel_replicate_degree=1,
    )
    job_config_training = TrainingConfig(
        seq_len=seq_len,
        local_batch_size=1,  # placeholder; we manage micro-batching ourselves
        global_batch_size=1,
        steps=1,
    )
    job_config_compile = CompileConfig(enable=False)  # compile interaction
    # with KD's two-model forward is fragile; off for now.
    job_config_ac = ActivationCheckpointConfig(mode="none")

    student = parallelize_kimi_linear(
        student,
        parallel_dims=parallel_dims,
        parallelism_config=job_config_parallelism,
        training_config=job_config_training,
        compile_config=job_config_compile,
        ac_config=job_config_ac,
    )

    # Materialize meta tensors on device + init.
    student.to_empty(device=device)
    spec.model.init_weights(student)

    return student, spec


# -----------------------------------------------------------------
# Checkpoint load (student)
# -----------------------------------------------------------------


def load_student_ckpt(student, ckpt_dir: str):
    """Load a torchtitan DCP checkpoint into the FSDP-sharded student.

    torchtitan saves under ``{out}/checkpoint/step-N``; we pass that
    full path. The state dict uses DCP sharded format.
    """
    state_dict = {"model": get_model_state_dict(student)}
    log(f"Loading student ckpt from {ckpt_dir}")
    dcp_load(state_dict, checkpoint_id=ckpt_dir)
    set_model_state_dict(
        student,
        state_dict["model"],
        options=StateDictOptions(
            full_state_dict=False,
            broadcast_from_rank0=False,
            strict=False,
        ),
    )
    log("Student ckpt loaded.")


def save_student_ckpt(student, ckpt_dir: str, step: int):
    """Save the student via DCP at ``{ckpt_dir}/checkpoint/step-{step}``."""
    target = Path(ckpt_dir) / "checkpoint" / f"step-{step}"
    target.mkdir(parents=True, exist_ok=True)
    state_dict = {"model": get_model_state_dict(student)}
    if is_rank0():
        log(f"Saving student ckpt to {target}")
    dcp_save(state_dict, checkpoint_id=str(target))


# -----------------------------------------------------------------
# Data
# -----------------------------------------------------------------


def build_dataloader(
    teacher_path: str,
    seq_len: int,
    local_bs: int,
    rank: int,
    world: int,
    seed: int,
):
    """Stream c4-en, tokenize with teacher's HF tokenizer (Kimi BPE),
    yield ``(input_ids, labels)`` batches sharded by rank.

    Implementation: HuggingFace ``datasets`` streaming + manual tokenize
    + simple chunking. Avoids torchtitan's HuggingFaceTextDataLoader
    because we want the exact same Kimi tokenizer as the teacher uses.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id or 0  # fallback

    ds = load_dataset(
        "allenai/c4", "en",
        split="train",
        streaming=True,
    )
    # Per-rank sharding via offset.
    ds = ds.shuffle(seed=seed + rank, buffer_size=1024)

    buf: list[int] = []
    while True:
        for sample in ds:
            ids = tokenizer.encode(sample["text"], add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos)
            # Emit as many full chunks as we can.
            while len(buf) >= (local_bs * (seq_len + 1)):
                chunk = buf[: local_bs * (seq_len + 1)]
                buf = buf[local_bs * (seq_len + 1):]
                t = torch.tensor(chunk, dtype=torch.long).view(local_bs, seq_len + 1)
                input_ids = t[:, :-1]
                labels = t[:, 1:]
                yield input_ids, labels


# -----------------------------------------------------------------
# Train loop
# -----------------------------------------------------------------


def main():
    args = parse_args()
    rank, world, device, mesh = setup_distributed()
    log(f"world={world}, device={device}, args={args}")

    torch.manual_seed(args.seed + rank)

    # Output dir.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- build student ----
    student, _spec = build_student(
        args.student_config, mesh, device, args.seq_len,
    )
    load_student_ckpt(student, args.student_ckpt)
    student.train()

    # ---- build teacher ----
    log(f"Loading teacher: {args.teacher}")
    teacher = TeacherRunner.load(
        args.teacher, device_mesh=mesh, dtype=torch.bfloat16,
        cache_dir=args.teacher_cache_dir,
    )
    log("Teacher loaded.")

    # ---- optimizer + scheduler ----
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    # Constant LR (post-cosine-decay distillation phase).
    sched = None

    # ---- KD config ----
    kd_cfg = KDConfig(
        alpha=args.kd_alpha,
        temperature=args.kd_temperature,
    )

    # ---- data ----
    dataloader = build_dataloader(
        args.teacher, args.seq_len, args.local_bs,
        rank=rank, world=world, seed=args.seed,
    )

    # ---- training loop ----
    grad_accum = max(1, args.global_bs // (args.local_bs * world))
    log(f"Grad accum steps: {grad_accum} "
        f"(global_bs={args.global_bs}, local_bs={args.local_bs}, world={world})")

    step = 0
    last_log = time.perf_counter()
    last_log_step = 0
    while step < args.steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            input_ids, labels = next(dataloader)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            student_logits = student(input_ids)  # [B, T, V]
            with torch.no_grad():
                teacher_logits = teacher(input_ids)  # [B, T, V]

            loss = kd_loss(student_logits, labels, teacher_logits, kd_cfg)
            # Sum-reduction in kd_loss; normalize by valid token count
            # so global loss is comparable to per-token CE.
            n_valid = (labels != kd_cfg.ignore_index).sum().clamp_min(1)
            loss = loss / n_valid / grad_accum
            loss.backward()
            accum_loss += loss.detach().item()

        # Gradient clipping at 1.0 (matches torchtitan default).
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optim.step()
        if sched is not None:
            sched.step()
        step += 1

        if step % args.log_freq == 0:
            now = time.perf_counter()
            tps = (
                (step - last_log_step) * args.local_bs * args.seq_len * grad_accum
                / max(now - last_log, 1e-6)
            )
            log(f"step {step:>6} / {args.steps} | "
                f"kd_loss={accum_loss:.4f} | "
                f"tps={tps:.0f} (per rank)")
            last_log = now
            last_log_step = step

        if step % args.save_freq == 0 or step == args.steps:
            save_student_ckpt(student, args.output_dir, step)

    log("KD training complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
