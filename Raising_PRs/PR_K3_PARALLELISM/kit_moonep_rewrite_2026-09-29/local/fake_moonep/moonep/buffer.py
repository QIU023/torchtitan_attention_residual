"""Stand-ins for moonep.buffer's pool allocation: a local [R * rows] tensor whose
chunk ``rank`` is the only one this process reads or writes."""

import torch


def pad_dim0_for_alignment(chunk_shape, dtype):
    # Pad like a VMM granularity would, so callers must slice the rows they use.
    return (chunk_shape[0] + 7) // 8 * 8 + 8


def create_nvl_dist_tensor(chunk_shape, dtype, local_rank, world_size, group=None):
    rows, *rest = chunk_shape
    return torch.full(
        (world_size * rows, *rest), float("nan"), dtype=dtype, device="cuda"
    ) if dtype.is_floating_point else torch.zeros(
        (world_size * rows, *rest), dtype=dtype, device="cuda"
    )
