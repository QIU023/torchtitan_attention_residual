"""A stand-in for MoonEP's public API (MoonshotAI/MoonEP 33327eb) for boxes without
NVSwitch multicast. It keeps the public signatures and the argument checks of
moonep/api.py, and implements the semantics with NCCL collectives on the EP group:

- plan: rank r's slot j holds expert ((r + 1) % R) * epn + j, so every slot is used;
  a token copy goes to its expert's home row when (token + k + src) is even, else to
  the slot that holds the expert;
- dispatch: rows ordered by VM group row (local experts, then slots), each non-empty
  group padded to token_padding with zero rows; cu_seqlens are padded group ends;
- combine: sums each token's K rows back on the source rank;
- prefetch_weight: fills this rank's slots from the owners' local rows;
- reduce_grad: adds every rank's slot gradients into the home rank's local rows and
  zeros the slots.

Only for validating an integration; never shipped.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from . import buffer  # noqa: F401

_ELEM_TYPES = (torch.bfloat16, torch.float32, torch.uint8)


@dataclass
class MoonEPCommPlan:
    experts_to_copy: torch.Tensor  # [R, epn] int32
    send_dst: torch.Tensor  # [S * K] destination rank of each (token, k)
    send_order: torch.Tensor  # [S * K] (token, k) indices sorted by destination
    send_splits: list[int]
    recv_splits: list[int]
    recv_rows: torch.Tensor  # [sum(recv_splits)] row in hidden_nvsh of each received copy
    N: int
    R: int
    E: int
    NvS: int
    K: int


def _validate_rank_strided_pool(pool: torch.Tensor) -> int:
    assert pool.is_cuda and pool.ndim >= 2 and pool.shape[0] > 0
    assert pool[0].is_contiguous(), "pools must be contiguous within each rank"
    rank_numel = pool[0].numel()
    rank_stride = int(pool.stride(0))
    assert rank_numel > 0 and rank_stride >= rank_numel, "rank payloads must not overlap"
    stride_bytes = rank_stride * pool.element_size()
    assert stride_bytes % 16 == 0 and 0 < stride_bytes < (1 << 40)
    assert pool.data_ptr() % 16 == 0, "pools must be 16-byte aligned"
    end = pool.storage_offset() + (pool.shape[0] - 1) * rank_stride + rank_numel
    assert end * pool.element_size() <= pool.untyped_storage().nbytes()
    return rank_stride


class Buffer:
    def __init__(self, S, H, K, E, num_ep_ranks, num_sms=None, token_padding=128, group=None):
        self.group = group
        self.R = num_ep_ranks
        assert self.R == dist.get_world_size(group)
        assert E % self.R == 0
        self.rank = dist.get_rank(group)
        self.S, self.H, self.K, self.E = S, H, K, E
        self.epn = E // self.R
        self.token_padding = token_padding
        self.N = S * K
        self.NvS = self.N * self.R + 2 * self.epn * token_padding
        self._destroyed = False

    def destroy(self):
        self._destroyed = True

    def _exchange(self, rows, send_splits, recv_splits):
        out = rows.new_empty(sum(recv_splits), *rows.shape[1:])
        dist.all_to_all_single(out, rows.contiguous(), recv_splits, send_splits, group=self.group)
        return out

    def _plan(self, topk_experts_sk):
        R, epn, rank = self.R, self.epn, self.rank
        dev = topk_experts_sk.device
        experts_to_copy = (
            ((torch.arange(R, device=dev)[:, None] + 1) % R) * epn
            + torch.arange(epn, device=dev)[None, :]
        ).to(torch.int32)
        e = topk_experts_sk.reshape(-1).long()
        home = e // epn
        t = torch.arange(self.S, device=dev).repeat_interleave(self.K)
        k = torch.arange(self.K, device=dev).repeat(self.S)
        to_home = (t + k + rank) % 2 == 0
        slot_rank = (home - 1) % R
        dst = torch.where(to_home, home, slot_rank)
        row_group = torch.where(to_home, e - home * epn, epn + (e - home * epn))
        order = torch.argsort(dst * (2 * epn) + row_group, stable=True)
        send_splits = torch.bincount(dst, minlength=R).tolist()
        recv = torch.empty(R, dtype=torch.long, device=dev)
        dist.all_to_all_single(recv, torch.tensor(send_splits, device=dev), group=self.group)
        recv_splits = recv.tolist()
        groups = self._exchange(row_group[order][:, None], send_splits, recv_splits)[:, 0]
        # Rows: group by VM row, then by source rank and token order within a group.
        by_group = torch.argsort(groups, stable=True)
        counts = torch.bincount(groups, minlength=2 * epn)
        padded = (counts + self.token_padding - 1) // self.token_padding * self.token_padding
        ends = torch.cumsum(padded, 0)
        starts = ends - padded
        rank_in_group = torch.empty_like(groups)
        sorted_groups = groups[by_group]
        first = torch.searchsorted(sorted_groups, torch.arange(2 * epn, device=dev))
        rank_in_group[by_group] = torch.arange(len(groups), device=dev) - first[sorted_groups]
        recv_rows = starts[groups] + rank_in_group
        assert int(ends[-1]) <= self.NvS
        plan = MoonEPCommPlan(
            experts_to_copy=experts_to_copy,
            send_dst=dst,
            send_order=order,
            send_splits=send_splits,
            recv_splits=recv_splits,
            recv_rows=recv_rows,
            N=self.N,
            R=R,
            E=self.E,
            NvS=self.NvS,
            K=self.K,
        )
        return plan, ends.to(torch.int32)

    def dispatch(
        self,
        hidden_sh,
        route_weights_sk=None,
        topk_experts_sk=None,
        tokens_per_expert=None,
        plan=None,
        async_finish=False,
        *,
        inter_rank_sync=True,
        zero_copy=False,
        router_weights_zero_copy=False,
    ):
        assert not self._destroyed and not async_finish and not zero_copy
        assert hidden_sh.dtype == torch.bfloat16 and hidden_sh.shape == (self.S, self.H)
        cu_seqlens = None
        if plan is None:
            assert topk_experts_sk is not None and tokens_per_expert is not None
            topk_flat = topk_experts_sk.reshape(-1)
            assert topk_flat.dtype == torch.int32 and topk_flat.numel() == self.N
            assert tokens_per_expert.dtype == torch.int32
            assert tokens_per_expert.numel() == self.E and tokens_per_expert.is_contiguous()
            plan, cu_seqlens = self._plan(topk_experts_sk)
        rows = hidden_sh.repeat_interleave(self.K, dim=0)[plan.send_order]
        recv = self._exchange(rows, plan.send_splits, plan.recv_splits)
        hidden_nvsh = hidden_sh.new_zeros(self.NvS, self.H)
        hidden_nvsh[plan.recv_rows] = recv
        weights_nvs = None
        if route_weights_sk is not None:
            assert route_weights_sk.dtype == torch.float32
            w = route_weights_sk.reshape(-1)[plan.send_order][:, None]
            recv_w = self._exchange(w, plan.send_splits, plan.recv_splits)[:, 0]
            weights_nvs = recv_w.new_zeros(self.NvS)
            weights_nvs[plan.recv_rows] = recv_w
        return hidden_nvsh, weights_nvs, cu_seqlens, plan

    def combine(
        self,
        plan=None,
        hidden_nvsh=None,
        route_weights_nvs=None,
        async_finish=False,
        *,
        inter_rank_sync=True,
        zero_copy=False,
        router_weights_zero_copy=False,
    ):
        assert not self._destroyed and not async_finish and not zero_copy
        assert hidden_nvsh.dtype == torch.bfloat16 and hidden_nvsh.shape == (self.NvS, self.H)
        back = self._exchange(hidden_nvsh[plan.recv_rows], plan.recv_splits, plan.send_splits)
        per_copy = torch.empty_like(back)
        per_copy[plan.send_order] = back
        output_sh = per_copy.view(self.S, self.K, self.H).float().sum(1).to(torch.bfloat16)
        gathered = None
        if route_weights_nvs is not None:
            w = route_weights_nvs[plan.recv_rows][:, None]
            back_w = self._exchange(w, plan.recv_splits, plan.send_splits)[:, 0]
            gathered = torch.empty_like(back_w)
            gathered[plan.send_order] = back_w
            gathered = gathered.view(self.S, self.K)
        return output_sh, gathered, None

    def prefetch_weight(
        self,
        plan=None,
        async_finish=False,
        *,
        local_gate_weight=None,
        local_up_weight=None,
        local_down_weight=None,
        gate_prefetch_buffer=None,
        up_prefetch_buffer=None,
        down_prefetch_buffer=None,
        **scales,
    ):
        assert isinstance(plan, MoonEPCommPlan) and not async_finish
        R, epn = self.R, self.epn
        local_weights = (local_gate_weight, local_up_weight, local_down_weight)
        prefetch_buffers = (gate_prefetch_buffer, up_prefetch_buffer, down_prefetch_buffer)
        assert all(t is not None for t in (*local_weights, *prefetch_buffers))
        for local, pool in zip(local_weights, prefetch_buffers, strict=True):
            assert local.dtype in _ELEM_TYPES
            assert local.is_contiguous()
            assert local.ndim == 3 and int(local.shape[0]) == epn
            assert pool.dtype == local.dtype and _validate_rank_strided_pool(pool)
            assert pool.ndim == 4 and tuple(pool.shape[:2]) == (R, epn)
            assert tuple(pool.shape[2:]) == tuple(local.shape[1:])
            everyone = [torch.empty_like(local) for _ in range(R)]
            dist.all_gather(everyone, local, group=self.group)
            for j, e in enumerate(plan.experts_to_copy[self.rank].tolist()):
                if e >= 0:
                    pool[self.rank, j].copy_(everyone[e // epn][e % epn])

    def reduce_grad(
        self,
        plan=None,
        async_finish=False,
        *,
        local_gate_grad=None,
        local_up_grad=None,
        local_down_grad=None,
        gate_reduce_buffer=None,
        up_reduce_buffer=None,
        down_reduce_buffer=None,
    ):
        assert isinstance(plan, MoonEPCommPlan) and not async_finish
        R, epn = self.R, self.epn
        locals_ = (local_gate_grad, local_up_grad, local_down_grad)
        pools = (gate_reduce_buffer, up_reduce_buffer, down_reduce_buffer)
        assert all(t is not None for t in (*locals_, *pools))
        for local, pool in zip(locals_, pools, strict=True):
            assert local.dtype == torch.float32 and local.is_contiguous()
            assert local.ndim == 3 and int(local.shape[0]) == epn
            assert pool.dtype == local.dtype and _validate_rank_strided_pool(pool)
            assert pool.ndim == 4 and tuple(pool.shape[:2]) == (R, epn)
            assert tuple(pool.shape[2:]) == tuple(local.shape[1:])
            mine = pool[self.rank].contiguous()
            everyone = [torch.empty_like(mine) for _ in range(R)]
            dist.all_gather(everyone, mine, group=self.group)
            for r in range(R):
                for j, e in enumerate(plan.experts_to_copy[r].tolist()):
                    if e >= 0 and e // epn == self.rank:
                        local[e % epn] += everyone[r][j]
            pool[self.rank].zero_()
