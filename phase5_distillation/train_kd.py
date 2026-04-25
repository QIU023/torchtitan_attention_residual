# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KD training entry point: 436M student × Kimi-Linear-48B-A3B-Base teacher.

Strategy: subclass torchtitan's Trainer to reuse the entire setup
(distributed, ParallelDims + mesh, model build via ModelSpec,
parallelize_kimi_linear, optimizer, scheduler, FSDP, dataloader,
checkpoint manager) and override only `forward_backward_step` to
swap the loss for the KD interpolation L = α·CE + (1-α)·T²·KL.

The teacher is built once after the student parallelize step,
FSDP2-sharded across the same DP mesh, kept in eval / no_grad. Each
microbatch goes through both models in the same training step;
backward only flows through the student. See
`docs/pretraining_closure_and_kd_plan.md` for the broader plan.

Run via launch_kd.sh.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure both the workspace and torchtitan submodule are importable.
WORKSPACE = Path(__file__).resolve().parent.parent
TORCHTITAN_PATH = WORKSPACE / "torchtitan"
for p in (str(WORKSPACE), str(TORCHTITAN_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torchtitan.trainer import Trainer  # noqa: E402
from torchtitan.tools.logging import init_logger, logger  # noqa: E402
import torchtitan.distributed.utils as dist_utils  # noqa: E402

from phase5_distillation.kd_loss import KDConfig, kd_loss  # noqa: E402
from phase5_distillation.teacher_runner import (  # noqa: E402
    DEFAULT_TEACHER, TeacherRunner,
)


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------


def _parse_kd_args() -> argparse.Namespace:
    """Pull KD-only flags out of sys.argv before torchtitan parses
    the rest. The rest (--module, --config, --training.*, etc.) is
    passed straight through to torchtitan.config.JobConfig.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--kd.teacher", default=DEFAULT_TEACHER,
                   dest="kd_teacher")
    p.add_argument("--kd.alpha", type=float, default=0.3,
                   dest="kd_alpha")
    p.add_argument("--kd.temperature", type=float, default=2.0,
                   dest="kd_temperature")
    p.add_argument("--kd.teacher-cache-dir", default=None,
                   dest="kd_teacher_cache_dir")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # leave the rest for torchtitan
    return args


# -----------------------------------------------------------------
# KD Trainer
# -----------------------------------------------------------------


class KDTrainer(Trainer):
    """torchtitan Trainer + teacher forward in `forward_backward_step`.

    Setup (everything Trainer.__init__ does) is unchanged. After
    super().__init__ finishes we have:
      - self.parallel_dims (ParallelDims with FSDP mesh)
      - self.model_parts (list, single FSDP-wrapped student here)
      - self.optimizers, self.lr_schedulers
      - self.dataloader, self.tokenizer
      - self.checkpointer
    All we add: a TeacherRunner on the same FSDP mesh + a KDConfig.
    """

    def __init__(self, config, *, teacher_path: str, kd_cfg: KDConfig,
                 teacher_cache_dir: str | None = None):
        super().__init__(config)

        # Teacher must use the same vocab as the student. We assume the
        # student's tokenizer was built from the teacher's HF assets
        # (passed via --hf_assets_path on the CLI). Sanity-check.
        if self.tokenizer is not None:
            student_vocab = getattr(self.tokenizer, "vocab_size", None)
            logger.info(
                f"KD: student tokenizer vocab_size={student_vocab}; "
                f"loading teacher {teacher_path} with same vocab assumed."
            )

        # Reuse the FSDP mesh that the student uses, so each rank's
        # data shard goes through both models on the same device.
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")

        logger.info(f"KD: loading teacher {teacher_path} ...")
        self.teacher = TeacherRunner.load(
            teacher_path,
            device_mesh=fsdp_mesh,
            dtype=torch.bfloat16,
            cache_dir=teacher_cache_dir,
        )
        logger.info("KD: teacher loaded.")

        self.kd_cfg = kd_cfg

    # -----------------------------------------------------------------
    # The single override.
    # -----------------------------------------------------------------
    def forward_backward_step(
        self,
        *,
        input_dict: dict,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """KD-interpolated loss. Mirrors Trainer.forward_backward_step
        for the non-PP path; PP-with-KD is out of scope here.
        """
        if self.parallel_dims.pp_enabled:
            raise NotImplementedError(
                "KD with PP is not implemented. Run KD on FSDP only."
            )

        inputs, labels_, extra_inputs, extra_kwargs = (
            self.post_dataloading_process(input_dict, labels)
        )

        student = self.model_parts[0]
        with self.train_context():
            student_logits = student(inputs, **extra_inputs, **extra_kwargs)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            loss_sum = kd_loss(
                student_logits, labels_, teacher_logits, self.kd_cfg,
            )
            loss = loss_sum / global_valid_tokens
            del student_logits, teacher_logits
            loss.backward()
        return loss

        return loss


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------


def main():
    init_logger()  # torchtitan's INFO->stdout handler; required because
    # we don't go through torchtitan/train.py main entry.
    kd_args = _parse_kd_args()

    # Have torchtitan parse the rest of sys.argv into a JobConfig.
    from torchtitan.config import ConfigManager

    config_manager = ConfigManager()
    # Pass sys.argv[1:] explicitly. ConfigManager.parse_args's default
    # arg is bound at function-definition time (i.e. when torchtitan
    # was imported, before _parse_kd_args modified sys.argv); without
    # this explicit pass tyro would still see the --kd.* flags.
    config = config_manager.parse_args(sys.argv[1:])

    # Ensure model_spec is set. ConfigManager.parse_args calls
    # `--module` -> registers ModelSpec on config; nothing extra to do.

    kd_cfg = KDConfig(
        alpha=kd_args.kd_alpha,
        temperature=kd_args.kd_temperature,
    )

    trainer = KDTrainer(
        config,
        teacher_path=kd_args.kd_teacher,
        kd_cfg=kd_cfg,
        teacher_cache_dir=kd_args.kd_teacher_cache_dir,
    )
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
