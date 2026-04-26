# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Continue pretraining the Phase 4 student on a MiniPLM-filtered c4 subset.

Pure CE loss. Reuses torchtitan's Trainer for everything (FSDP, optim,
scheduler, ckpt) and swaps only the dataloader to read pre-tokenized
chunks from a local filtered.jsonl.

Lives outside the torchtitan submodule on purpose. Custom dataloader
uses torchdata's StatefulDataLoader (the same wrapper torchtitan's
HuggingFaceTextDataLoader uses) so torch.compile sees the SAME tensor
contract (dtype, device, stride layout) torchtitan's main path
provides — without that wrapper, compile silently re-traces every
batch and stalls indefinitely.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

WORKSPACE = Path(__file__).resolve().parent.parent.parent
TORCHTITAN_PATH = WORKSPACE / "torchtitan"
for p in (str(WORKSPACE), str(TORCHTITAN_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torchdata.stateful_dataloader import StatefulDataLoader  # noqa: E402

from torchtitan.trainer import Trainer  # noqa: E402
from torchtitan.tools.logging import init_logger, logger  # noqa: E402
from torchtitan.components.dataloader import ParallelAwareDataloader  # noqa: E402


# -----------------------------------------------------------------
# Per-example IterableDataset (no batching here — DataLoader handles it)
# -----------------------------------------------------------------


class LocalJsonlIterableDataset(IterableDataset):
    """Streams pre-tokenized chunks from a local .jsonl, one example
    per yield. Sharded by (dp_rank, world_size) and then by DataLoader
    worker index so multiple workers per rank don't read the same line.

    Each yielded record:
        ({"input": LongTensor[seq_len]}, LongTensor[seq_len])

    Yields infinitely (loops over the file).
    """

    def __init__(self, jsonl_path: str, dp_rank: int, dp_world_size: int,
                 seq_len: int):
        self.path = jsonl_path
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(jsonl_path)

    def __iter__(self):
        wi = get_worker_info()
        # Combine DP-rank stride with per-worker stride so M workers
        # across N DP ranks each read a unique 1/(M*N) shard of the file.
        if wi is not None:
            local_stride = wi.num_workers
            local_offset = wi.id
        else:
            local_stride = 1
            local_offset = 0
        total_stride = self.dp_world_size * local_stride
        my_offset = self.dp_rank * local_stride + local_offset

        # score_corpus.py packed chunks at exactly seq_len tokens; we
        # need seq_len+1 to make a shifted (input, label) pair.
        # Solution: treat the jsonl as a token stream, re-window across
        # chunk boundaries into seq_len+1 windows. Same pattern as
        # torchtitan's HuggingFaceTextDataset (cross-document concat +
        # window). Zero tokens lost; cross-chunk boundaries are
        # acceptable because the original chunks were already
        # cross-document concatenations from c4 streaming.
        win = self.seq_len + 1
        buf: list[int] = []
        while True:
            with open(self.path, "r") as f:
                for line_idx, line in enumerate(f):
                    if line_idx % total_stride != my_offset:
                        continue
                    rec = json.loads(line)
                    buf.extend(rec["input_ids"])
                    while len(buf) >= win:
                        chunk = torch.tensor(buf[:win], dtype=torch.long)
                        buf = buf[win:]
                        yield ({"input": chunk[:-1]}, chunk[1:])

    # Stateful protocol — minimal no-op (continued pretraining doesn't
    # need precise dataloader resume; ckpt resumes model+optim only).
    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


def _collate(batch):
    """Collate a list of ({"input": T[L]}, T[L]) into ({"input": T[B,L]}, T[B,L])."""
    inputs = torch.stack([b[0]["input"] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return ({"input": inputs}, labels)


# -----------------------------------------------------------------
# Trainer subclass — only override is replacing self.dataloader
# -----------------------------------------------------------------


class FilteredCETrainer(Trainer):
    def __init__(self, config, *, filtered_jsonl: str,
                 num_workers: int = 0, prefetch_factor: int = 2):
        # num_workers=0 default: DataLoader runs in the main process.
        # workers > 0 forks after Trainer.__init__ has already initialized
        # NCCL, so the child workers inherit a corrupted CUDA context and
        # the next NCCL collective deadlocks. Phase 4's main-path
        # HuggingFaceTextDataLoader uses num_workers=0 for the same
        # reason. Override at the CLI only if you have a fix for the
        # fork-after-CUDA-init issue.
        super().__init__(config)

        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            dp_world_size = batch_mesh.size()
            dp_rank = batch_mesh.get_local_rank()
        else:
            dp_world_size, dp_rank = 1, 0

        ds = LocalJsonlIterableDataset(
            jsonl_path=filtered_jsonl,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seq_len=config.training.seq_len,
        )
        # Wrap in the same StatefulDataLoader wrapper torchtitan uses,
        # giving us multi-worker prefetch + proper batching contract for
        # torch.compile (compile cache hit relies on stable
        # dtype/device/stride from the dataloader).
        loader = ParallelAwareDataloader(
            ds,                           # positional — _validate_kwargs forbids kw form
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=config.training.local_batch_size,
            collate_fn=_collate,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=(num_workers > 0),
        )
        logger.info(
            f"miniplm: replaced dataloader with ParallelAwareDataloader "
            f"(jsonl={filtered_jsonl}, dp_rank={dp_rank}/{dp_world_size}, "
            f"workers={num_workers}, prefetch={prefetch_factor}, "
            f"local_bs={config.training.local_batch_size})"
        )
        self.dataloader = loader


def main():
    init_logger()

    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--miniplm.filtered", dest="miniplm_filtered", required=True)
    p.add_argument("--miniplm.num-workers", type=int, default=0,
                   dest="miniplm_num_workers")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    from torchtitan.config import ConfigManager
    cm = ConfigManager()
    config = cm.parse_args(sys.argv[1:])

    trainer = FilteredCETrainer(
        config,
        filtered_jsonl=args.miniplm_filtered,
        num_workers=args.miniplm_num_workers,
    )
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
