# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Continue pretraining the Phase 4 student on a MiniPLM-filtered c4 subset.

Pure CE loss — NO teacher forward in the train loop. The teacher's
contribution is already baked into the filtered corpus via score_corpus.py
+ filter_corpus.py.

Reuses torchtitan's Trainer for everything (FSDP, optim, scheduler,
ckpt) and only swaps the dataloader to read pre-tokenized chunks from
a local filtered.jsonl. No torchtitan submodule changes — the
custom dataloader lives entirely in this file.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

WORKSPACE = Path(__file__).resolve().parent.parent.parent
TORCHTITAN_PATH = WORKSPACE / "torchtitan"
for p in (str(WORKSPACE), str(TORCHTITAN_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torchtitan.trainer import Trainer  # noqa: E402
from torchtitan.tools.logging import init_logger, logger  # noqa: E402


# -----------------------------------------------------------------
# Filtered jsonl dataloader (drop-in for torchtitan dataloader)
# -----------------------------------------------------------------


class FilteredJsonlDataset(IterableDataset):
    """Streams pre-tokenized chunks from a jsonl file.

    Each line is `{"input_ids": [int, int, ...]}` of length seq_len+1
    (we'll yield (input_ids[:-1], input_ids[1:]) as the (x, y) pair).
    Sharded across DP ranks by line index modulo dp_world_size, then
    looped infinitely for continuous training.
    """

    def __init__(self, jsonl_path: str, dp_rank: int, dp_world_size: int,
                 seq_len: int, local_batch_size: int):
        self.path = jsonl_path
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        # Validate file exists.
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(jsonl_path)

    def _stream_one_pass(self):
        with open(self.path, "r") as f:
            for line_idx, line in enumerate(f):
                if line_idx % self.dp_world_size != self.dp_rank:
                    continue
                rec = json.loads(line)
                ids = rec["input_ids"]
                if len(ids) < self.seq_len + 1:
                    continue  # skip too-short chunks
                yield ids[: self.seq_len + 1]

    def __iter__(self):
        # Infinite loop — torchtitan's Trainer drives stop via training.steps
        while True:
            buf: list[list[int]] = []
            for ids in self._stream_one_pass():
                buf.append(ids)
                if len(buf) >= self.local_batch_size:
                    chunk = torch.tensor(buf[: self.local_batch_size], dtype=torch.long)
                    buf = buf[self.local_batch_size:]
                    yield (
                        {"input": chunk[:, :-1]},  # input_dict matches torchtitan trainer expectation
                        chunk[:, 1:],              # labels
                    )

    # torchtitan's Stateful protocol — dataloader resume isn't critical
    # for continued pretraining (we just keep streaming), so no-op state.
    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


class FilteredDataLoader:
    """Wraps FilteredJsonlDataset to look like torchtitan's BaseDataLoader
    (only the methods Trainer actually calls).
    """

    def __init__(self, dataset: FilteredJsonlDataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


# -----------------------------------------------------------------
# Trainer subclass that swaps the dataloader after super().__init__
# -----------------------------------------------------------------


class FilteredCETrainer(Trainer):
    """Standard CE training on a local filtered.jsonl. The only override
    is replacing self.dataloader after Trainer's normal setup; loss,
    forward_backward_step, and everything else use Trainer defaults.
    """

    def __init__(self, config, *, filtered_jsonl: str):
        super().__init__(config)

        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            dp_world_size = batch_mesh.size()
            dp_rank = batch_mesh.get_local_rank()
        else:
            dp_world_size, dp_rank = 1, 0

        logger.info(
            f"miniplm: replacing dataloader with FilteredJsonlDataset "
            f"(path={filtered_jsonl}, dp_rank={dp_rank}/{dp_world_size}, "
            f"seq_len={config.training.seq_len}, "
            f"local_bs={config.training.local_batch_size})"
        )
        ds = FilteredJsonlDataset(
            jsonl_path=filtered_jsonl,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
        )
        self.dataloader = FilteredDataLoader(ds)


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------


def main():
    init_logger()

    # Pull --miniplm.filtered out of argv before torchtitan's tyro parser.
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--miniplm.filtered", dest="miniplm_filtered", required=True,
                   help="Path to filtered.jsonl from filter_corpus.py")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    from torchtitan.config import ConfigManager
    cm = ConfigManager()
    config = cm.parse_args(sys.argv[1:])

    trainer = FilteredCETrainer(config, filtered_jsonl=args.miniplm_filtered)
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
