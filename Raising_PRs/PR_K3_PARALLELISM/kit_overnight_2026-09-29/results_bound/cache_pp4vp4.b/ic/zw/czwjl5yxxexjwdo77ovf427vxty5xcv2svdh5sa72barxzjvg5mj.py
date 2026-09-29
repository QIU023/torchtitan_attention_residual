# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/dk/cdkhdlmtno5zacd65njj2og63mejq4aogk44a7sxzfzbedckaknd.py
# Topologically Sorted Source Nodes: [m, index, batched_outputs, n, index_1, batched_outputs_3, mask, mask_1, mask_2, mask_3, mask_block_sum, gt, lt, partial_blocks, partial_blocks_1, dense_mask, full_blocks, full_blocks_1, dense_mask_1], Original ATen: [aten.arange, aten.index, aten.view, aten.eq, aten.unsqueeze, aten.constant_pad_nd, aten.permute, aten.sum, aten.gt, aten.lt, aten.bitwise_and, aten._to_copy]
# Source node to ATen node mapping:
#   batched_outputs => eq, view
#   batched_outputs_3 => unsqueeze
#   dense_mask => convert_element_type_2
#   dense_mask_1 => convert_element_type_5
#   full_blocks => eq_1
#   full_blocks_1 => convert_element_type_1
#   gt => gt
#   index => index
#   index_1 => index_1
#   lt => lt
#   m => iota_2
#   mask => unsqueeze_1
#   mask_1 => constant_pad_nd
#   mask_2 => view_1
#   mask_3 => permute
#   mask_block_sum => sum_1
#   n => iota_3
#   partial_blocks => bitwise_and
#   partial_blocks_1 => convert_element_type
# Graph fragment:
#   %arg2_1 : Tensor "i32[4][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %sum_1 : Tensor "i64[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][(((s12 + 127)//128))*(((s37 + 127)//128)), (((s12 + 127)//128))*(((s37 + 127)//128)), ((s37 + 127)//128), 1]cuda:0" = PlaceHolder[target=sum_1]
#   %iota_2 : Tensor "i64[s12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %index : Tensor "i32[s12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%iota_2]), kwargs = {})
#   %view : Tensor "i32[s12, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [%arg0_1, 1]), kwargs = {})
#   %iota_3 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg1_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %index_1 : Tensor "i32[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%iota_3]), kwargs = {})
#   %eq : Tensor "b8[s12, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view, %index_1), kwargs = {})
#   %unsqueeze : Tensor "b8[1, s12, s37][s12*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "b8[1, 1, s12, s37][s12*Max(1, s37), s12*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%expand, 0), kwargs = {})
#   %constant_pad_nd : Tensor "b8[1, 1, 128*(((s12 + 127)//128)), 128*(((s37 + 127)//128))][Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s37 + 127)//128))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%expand_1, [0, %sub_5, 0, %sub_7], 0.0), kwargs = {})
#   %view_1 : Tensor "b8[1, 1, ((s12 + 127)//128), 128, ((s37 + 127)//128), 128][Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), 128*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s37 + 127)//128))), 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [1, 1, %floordiv_3, 128, %floordiv_2, 128]), kwargs = {})
#   %permute : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128), 128, 128][Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), 128*Max(1, 128*(((s37 + 127)//128))), 128, Max(1, 128*(((s37 + 127)//128))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 1, 2, 4, 3, 5]), kwargs = {})
#   %sum_1 : Tensor "i64[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute, [-2, -1]), kwargs = {})
#   %gt : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_1, 0), kwargs = {})
#   %lt : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %bitwise_and : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt, %lt), kwargs = {})
#   %convert_element_type : Tensor "i8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and, torch.int8), kwargs = {})
#   %convert_element_type_2 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.int32), kwargs = {})
#   %eq_1 : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %convert_element_type_1 : Tensor "i8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq_1, torch.int8), kwargs = {})
#   %convert_element_type_5 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.int32), kwargs = {})
#   return %sum_1,%convert_element_type_2,%convert_element_type_5
triton_red_fused__to_copy_arange_bitwise_and_constant_pad_nd_eq_gt_index_lt_permute_sum_unsqueeze_view_0 = async_compile.triton('triton_red_fused__to_copy_arange_bitwise_and_constant_pad_nd_eq_gt_index_lt_permute_sum_unsqueeze_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_arange_bitwise_and_constant_pad_nd_eq_gt_index_lt_permute_sum_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'r0_': 16}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_arange_bitwise_and_constant_pad_nd_eq_gt_index_lt_permute_sum_unsqueeze_view_0(in_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 128
        r0_2 = (r0_index % 128)
        tmp0 = (r0_3*(triton_helpers.div_floor_integer(127 + ks0,  128))).to(tl.int32)
        tmp1 = (ks1).to(tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r0_2).to(tl.int32)
        tmp4 = (ks0).to(tl.int32)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp2 & tmp5
        tl.device_assert((tl.broadcast_to(r0_3*(triton_helpers.div_floor_integer(127 + ks0,  128)), [XBLOCK, R0_BLOCK]) < 4) | ~(r0_mask & tmp6), "index out of bounds: tl.broadcast_to(r0_3*(triton_helpers.div_floor_integer(127 + ks0,  128)), [XBLOCK, R0_BLOCK]) < 4")
        tmp8 = tl.load(in_ptr0 + (tl.broadcast_to(r0_3*(triton_helpers.div_floor_integer(127 + ks0,  128)), [XBLOCK, R0_BLOCK])), r0_mask & tmp6, eviction_policy='evict_last', other=0.0)
        tl.device_assert((tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK]) < 4) | ~(r0_mask & tmp6), "index out of bounds: tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK]) < 4")
        tmp10 = tl.load(in_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp6, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp8 == tmp10
        tmp12 = tl.full(tmp11.shape, False, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp13.to(tl.int64)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp18 = tl.full([1, 1], 0, tl.int64)
    tmp19 = tmp16 > tmp18
    tmp20 = tl.full([1, 1], 16384, tl.int64)
    tmp21 = tmp16 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22.to(tl.int8)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tmp16 == tmp20
    tmp26 = tmp25.to(tl.int8)
    tmp27 = tmp26.to(tl.int32)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32)), tmp24, None)
    tl.store(out_ptr2 + (tl.full([1, 1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/a4/ca4emf3wgl6h3mjvcmxm5xqizt66yf2cvxfh5ep5v44nmezidtww.py
# Topologically Sorted Source Nodes: [num_blocks_in_row, child_3], Original ATen: [aten.sum, aten._to_copy]
# Source node to ATen node mapping:
#   child_3 => convert_element_type_3
#   num_blocks_in_row => sum_2
# Graph fragment:
#   %convert_element_type_2 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][(((s12 + 127)//128))*(((s37 + 127)//128)), (((s12 + 127)//128))*(((s37 + 127)//128)), ((s37 + 127)//128), 1]cuda:0" = PlaceHolder[target=convert_element_type_2]
#   %sum_2 : Tensor "i64[1, 1, ((s12 + 127)//128)][((s12 + 127)//128), ((s12 + 127)//128), 1]cuda:0" = PlaceHolder[target=sum_2]
#   %sum_2 : Tensor "i64[1, 1, ((s12 + 127)//128)][Max(1, ((s12 + 127)//128)), Max(1, ((s12 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_2, [-1]), kwargs = {})
#   %convert_element_type_3 : Tensor "i32[1, 1, ((s12 + 127)//128)][Max(1, ((s12 + 127)//128)), Max(1, ((s12 + 127)//128)), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.int32), kwargs = {})
#   return %sum_2,%convert_element_type_3
triton_per_fused__to_copy_sum_1 = async_compile.triton('triton_per_fused__to_copy_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'r0_': 0}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__to_copy_sum_1(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    tmp0 = tl.load(in_ptr0 + (tl.full([1, 1], 0, tl.int32)), None, eviction_policy='evict_first')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.int64)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32)), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/sw/cswvqee4qz3lv5wbulg6kvjmedqrvrgo4u3hteovyyc7zv5krbob.py
# Topologically Sorted Source Nodes: [child_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   child_4 => convert_element_type_4
# Graph fragment:
#   %getitem_1 : Tensor "i64[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %convert_element_type_4 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1, torch.int32), kwargs = {})
#   return %convert_element_type_4
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 0}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int32)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/ka/ckay3whctcc3ggb5vgotj46riniy3qz3cai7jbeyxtfpkivvozis.py
# Topologically Sorted Source Nodes: [dense_mask_2], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   dense_mask_2 => full_default
# Graph fragment:
#   %full_default : Tensor "i32[1, 1, ((s12 + 127)//128), (((s37 + 127)//128)) + 1][Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, %floordiv_3, %add_4], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %index_put
triton_poi_fused_new_zeros_3 = async_compile.triton('triton_poi_fused_new_zeros_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_new_zeros_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 4}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/zz/czzvx4hysgwgorznybkkr5ux4i43a6rdzi4ue2era55p32lslcnd.py
# Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   setitem => full_default_1
# Graph fragment:
#   %full_default_1 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1, 1], 1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default_1
triton_poi_fused_view_4 = async_compile.triton('triton_poi_fused_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.full([1], 1, tl.int32)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/2t/c2tztsiscmlqj2gxs5nb4qmika2h2mxxhlxqzcs3ydzeua6oxwyj.py
# Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   setitem => iota_7
# Graph fragment:
#   %iota_7 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   return %iota_7
triton_poi_fused_arange_5 = async_compile.triton('triton_poi_fused_arange_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/oc/cocxoqgmqju5jgen6yrlehlufgb3vmmpiucyyhrylum6pw5uyftt.py
# Topologically Sorted Source Nodes: [arange_4], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   arange_4 => iota_4
# Graph fragment:
#   %iota_4 : Tensor "i32[((s12 + 127)//128)][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_3,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:0, requires_grad: False})
#   return %iota_4
triton_poi_fused_arange_6 = async_compile.triton('triton_poi_fused_arange_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 0}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = (tl.full([XBLOCK], 0, tl.int32)).to(tl.int32)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/bv/cbvir3od5pguqi44mbdvrwrwf6cfejj2cd3pvpnkzi4g2a7jugxu.py
# Topologically Sorted Source Nodes: [col_range, unsqueeze_1, index_mask, valid_indices], Original ATen: [aten.arange, aten.unsqueeze, aten.lt, aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   col_range => iota_5
#   index_mask => lt_1
#   unsqueeze_1 => unsqueeze_3
#   valid_indices => scalar_tensor, where
# Graph fragment:
#   %convert_element_type_3 : Tensor "i32[1, 1, ((s12 + 127)//128)][Max(1, ((s12 + 127)//128)), Max(1, ((s12 + 127)//128)), 1]cuda:0" = PlaceHolder[target=convert_element_type_3]
#   %convert_element_type_4 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0" = PlaceHolder[target=convert_element_type_4]
#   %iota_5 : Tensor "i32[((s37 + 127)//128)][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_2,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:0, requires_grad: False})
#   %unsqueeze_3 : Tensor "i32[1, 1, ((s12 + 127)//128), 1][Max(1, ((s12 + 127)//128)), Max(1, ((s12 + 127)//128)), 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 3), kwargs = {})
#   %lt_1 : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_5, %unsqueeze_3), kwargs = {})
#   %scalar_tensor : Tensor "i32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%floordiv_2,), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0})
#   %where : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_1, %convert_element_type_4, %scalar_tensor), kwargs = {})
#   return %where
triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7 = async_compile.triton('triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*i32', 'out_ptr0': '*i32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 0}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), None)
    tmp2 = (tl.full([XBLOCK], 0, tl.int32)).to(tl.int32)
    tmp3 = tmp2 < tmp1
    tmp5 = (triton_helpers.div_floor_integer(127 + ks0,  128)).to(tl.int32)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        s12 = arg0_1
        s37 = arg1_1
        assert_size_stride(arg2_1, (4, ), (1, ), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf1 = empty_strided_cuda((1, 1, (127 + s12) // 128, (127 + s37) // 128), (((127 + s12) // 128)*((127 + s37) // 128), ((127 + s12) // 128)*((127 + s37) // 128), (127 + s37) // 128, 1), torch.int32)
            buf5 = empty_strided_cuda((1, 1, (127 + s12) // 128, (127 + s37) // 128), (((127 + s12) // 128)*((127 + s37) // 128), ((127 + s12) // 128)*((127 + s37) // 128), (127 + s37) // 128, 1), torch.int32)
            # Topologically Sorted Source Nodes: [m, index, batched_outputs, n, index_1, batched_outputs_3, mask, mask_1, mask_2, mask_3, mask_block_sum, gt, lt, partial_blocks, partial_blocks_1, dense_mask, full_blocks, full_blocks_1, dense_mask_1], Original ATen: [aten.arange, aten.index, aten.view, aten.eq, aten.unsqueeze, aten.constant_pad_nd, aten.permute, aten.sum, aten.gt, aten.lt, aten.bitwise_and, aten._to_copy]
            triton_red_fused__to_copy_arange_bitwise_and_constant_pad_nd_eq_gt_index_lt_permute_sum_unsqueeze_view_0_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_arange_bitwise_and_constant_pad_nd_eq_gt_index_lt_permute_sum_unsqueeze_view_0.run(arg2_1, buf1, buf5, s37, s12, triton_red_fused__to_copy_arange_bitwise_and_constant_pad_nd_eq_gt_index_lt_permute_sum_unsqueeze_view_0_xnumel, 16384, stream=raw_stream0)
            del arg2_1
            # Topologically Sorted Source Nodes: [gt, lt, partial_blocks, partial_blocks_1, dense_mask, col_indices], Original ATen: [aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.sort]
            buf2 = torch.ops.aten.sort.stable(buf1, stable=True, descending=True)
            buf4 = buf2[1]
            assert_tensor_metadata(buf4, (1, 1, (127 + s12) // 128, (127 + s37) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int64, 'torch.ops.aten.sort.stable')
            assert_alignment(buf4, 16, 'torch.ops.aten.sort.stable')
            del buf2
            # Topologically Sorted Source Nodes: [full_blocks, full_blocks_1, dense_mask_1, col_indices_1], Original ATen: [aten.eq, aten._to_copy, aten.sort]
            buf6 = torch.ops.aten.sort.stable(buf5, stable=True, descending=True)
            buf8 = buf6[1]
            assert_tensor_metadata(buf8, (1, 1, (127 + s12) // 128, (127 + s37) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int64, 'torch.ops.aten.sort.stable')
            assert_alignment(buf8, 16, 'torch.ops.aten.sort.stable')
            del buf6
            buf10 = empty_strided_cuda((1, 1, (127 + s12) // 128), (max(1, (127 + s12) // 128), max(1, (127 + s12) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [num_blocks_in_row, child_3], Original ATen: [aten.sum, aten._to_copy]
            triton_per_fused__to_copy_sum_1_xnumel = (127 + s12) // 128
            triton_per_fused__to_copy_sum_1_r0_numel = (127 + s37) // 128
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_sum_1.run(buf1, buf10, triton_per_fused__to_copy_sum_1_xnumel, triton_per_fused__to_copy_sum_1_r0_numel, stream=raw_stream0)
            del buf1
            buf11 = empty_strided_cuda((1, 1, (127 + s12) // 128, (127 + s37) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [child_4], Original ATen: [aten._to_copy]
            triton_poi_fused__to_copy_2_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf4, buf11, triton_poi_fused__to_copy_2_xnumel, stream=raw_stream0)
            del buf4
            buf12 = empty_strided_cuda((1, 1, (127 + s12) // 128, 1 + ((127 + s37) // 128)), (((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), 1 + ((127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_2], Original ATen: [aten.new_zeros]
            triton_poi_fused_new_zeros_3_xnumel = ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_3.run(buf12, triton_poi_fused_new_zeros_3_xnumel, stream=raw_stream0)
            buf13 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
            # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_view_4.run(buf13, 1, stream=raw_stream0)
            buf14 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_5.run(buf14, 1, stream=raw_stream0)
            buf15 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_5.run(buf15, 1, stream=raw_stream0)
            buf16 = empty_strided_cuda(((127 + s12) // 128, ), (1, ), torch.int32)
            # Topologically Sorted Source Nodes: [arange_4], Original ATen: [aten.arange]
            triton_poi_fused_arange_6_xnumel = (127 + s12) // 128
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_6.run(buf16, triton_poi_fused_arange_6_xnumel, stream=raw_stream0)
            buf17 = empty_strided_cuda((1, 1, 1, (127 + s37) // 128), ((127 + s37) // 128, (127 + s37) // 128, (127 + s37) // 128, 1), torch.int32)
            # Topologically Sorted Source Nodes: [col_range, unsqueeze_1, index_mask, valid_indices], Original ATen: [aten.arange, aten.unsqueeze, aten.lt, aten.scalar_tensor, aten.where]
            triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7_xnumel = (127 + s37) // 128
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7.run(buf10, buf11, buf17, s37, triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7_xnumel, stream=raw_stream0)
            aten.index_put_(buf12, [reinterpret_tensor(buf14, (1, 1, 1, 1), (0, 0, 0, 0), 0), reinterpret_tensor(buf15, (1, 1, 1), (0, 0, 0), 0), reinterpret_tensor(buf16, ((127 + s12) // 128, 1), (1, 0), 0), buf17], buf13, False)
            del buf17
            # Topologically Sorted Source Nodes: [num_blocks_in_row_2, col_indices_2], Original ATen: [aten.slice, aten.transpose, aten.sort]
            buf19 = torch.ops.aten.sort.stable(reinterpret_tensor(buf12, (1, 1, (127 + s37) // 128, (127 + s12) // 128), (((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), 1, 1 + ((127 + s37) // 128)), 0), stable=True, descending=True)
            buf21 = buf19[1]
            assert_tensor_metadata(buf21, (1, 1, (127 + s37) // 128, (127 + s12) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), 1, max(1, (127 + s37) // 128)), torch.int64, 'torch.ops.aten.sort.stable')
            assert_alignment(buf21, 16, 'torch.ops.aten.sort.stable')
            del buf19
            buf23 = reinterpret_tensor(buf16, (1, 1, (127 + s12) // 128), (max(1, (127 + s12) // 128), max(1, (127 + s12) // 128), 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [num_blocks_in_row_1, child_7], Original ATen: [aten.sum, aten._to_copy]
            triton_per_fused__to_copy_sum_1_xnumel = (127 + s12) // 128
            triton_per_fused__to_copy_sum_1_r0_numel = (127 + s37) // 128
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_sum_1.run(buf5, buf23, triton_per_fused__to_copy_sum_1_xnumel, triton_per_fused__to_copy_sum_1_r0_numel, stream=raw_stream0)
            del buf5
            buf24 = empty_strided_cuda((1, 1, (127 + s12) // 128, (127 + s37) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [child_8], Original ATen: [aten._to_copy]
            triton_poi_fused__to_copy_2_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf8, buf24, triton_poi_fused__to_copy_2_xnumel, stream=raw_stream0)
            del buf8
            buf25 = empty_strided_cuda((1, 1, (127 + s12) // 128, 1 + ((127 + s37) // 128)), (((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), 1 + ((127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_4], Original ATen: [aten.new_zeros]
            triton_poi_fused_new_zeros_3_xnumel = ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_3.run(buf25, triton_poi_fused_new_zeros_3_xnumel, stream=raw_stream0)
            buf26 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_view_4.run(buf26, 1, stream=raw_stream0)
            buf27 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_5.run(buf27, 1, stream=raw_stream0)
            buf28 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.arange]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_5.run(buf28, 1, stream=raw_stream0)
            buf29 = empty_strided_cuda(((127 + s12) // 128, ), (1, ), torch.int32)
            # Topologically Sorted Source Nodes: [arange_6], Original ATen: [aten.arange]
            triton_poi_fused_arange_6_xnumel = (127 + s12) // 128
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_6.run(buf29, triton_poi_fused_arange_6_xnumel, stream=raw_stream0)
            buf30 = empty_strided_cuda((1, 1, 1, (127 + s37) // 128), ((127 + s37) // 128, (127 + s37) // 128, (127 + s37) // 128, 1), torch.int32)
            # Topologically Sorted Source Nodes: [col_range_1, unsqueeze_3, index_mask_1, valid_indices_1], Original ATen: [aten.arange, aten.unsqueeze, aten.lt, aten.scalar_tensor, aten.where]
            triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7_xnumel = (127 + s37) // 128
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7.run(buf23, buf24, buf30, s37, triton_poi_fused_arange_lt_scalar_tensor_unsqueeze_where_7_xnumel, stream=raw_stream0)
            aten.index_put_(buf25, [reinterpret_tensor(buf27, (1, 1, 1, 1), (0, 0, 0, 0), 0), reinterpret_tensor(buf28, (1, 1, 1), (0, 0, 0), 0), reinterpret_tensor(buf29, ((127 + s12) // 128, 1), (1, 0), 0), buf30], buf26, False)
            del buf26
            del buf27
            del buf28
            del buf29
            # Topologically Sorted Source Nodes: [num_blocks_in_row_3, col_indices_3], Original ATen: [aten.slice, aten.transpose, aten.sort]
            buf32 = torch.ops.aten.sort.stable(reinterpret_tensor(buf25, (1, 1, (127 + s37) // 128, (127 + s12) // 128), (((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), 1, 1 + ((127 + s37) // 128)), 0), stable=True, descending=True)
            buf34 = buf32[1]
            assert_tensor_metadata(buf34, (1, 1, (127 + s37) // 128, (127 + s12) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), 1, max(1, (127 + s37) // 128)), torch.int64, 'torch.ops.aten.sort.stable')
            assert_alignment(buf34, 16, 'torch.ops.aten.sort.stable')
            del buf32
            buf35 = empty_strided_cuda((1, 1, (127 + s37) // 128, (127 + s12) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [full_q_indices], Original ATen: [aten._to_copy]
            triton_poi_fused__to_copy_2_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf34, buf35, triton_poi_fused__to_copy_2_xnumel, stream=raw_stream0)
            del buf34
            buf37 = reinterpret_tensor(buf30, (1, 1, (127 + s37) // 128), (max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [num_blocks_in_row_3, full_q_num_blocks], Original ATen: [aten.slice, aten.transpose, aten.sum, aten._to_copy]
            triton_per_fused__to_copy_sum_1_xnumel = (127 + s37) // 128
            triton_per_fused__to_copy_sum_1_r0_numel = (127 + s12) // 128
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_sum_1.run(buf25, buf37, triton_per_fused__to_copy_sum_1_xnumel, triton_per_fused__to_copy_sum_1_r0_numel, stream=raw_stream0)
            del buf25
            buf38 = empty_strided_cuda((1, 1, (127 + s37) // 128, (127 + s12) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [q_indices], Original ATen: [aten._to_copy]
            triton_poi_fused__to_copy_2_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf21, buf38, triton_poi_fused__to_copy_2_xnumel, stream=raw_stream0)
            del buf21
            buf40 = empty_strided_cuda((1, 1, (127 + s37) // 128), (max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [num_blocks_in_row_2, q_num_blocks], Original ATen: [aten.slice, aten.transpose, aten.sum, aten._to_copy]
            triton_per_fused__to_copy_sum_1_xnumel = (127 + s37) // 128
            triton_per_fused__to_copy_sum_1_r0_numel = (127 + s12) // 128
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_sum_1.run(buf12, buf40, triton_per_fused__to_copy_sum_1_xnumel, triton_per_fused__to_copy_sum_1_r0_numel, stream=raw_stream0)
            del buf12
        return (buf35, buf37, buf38, buf40, buf24, buf23, buf11, buf10, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 4
    arg1_1 = 4
    arg2_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int32)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
