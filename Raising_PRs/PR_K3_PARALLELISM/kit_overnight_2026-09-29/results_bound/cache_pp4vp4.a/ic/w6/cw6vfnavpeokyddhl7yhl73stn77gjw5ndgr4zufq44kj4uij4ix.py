# AOT ID: ['0_inference']
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


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/cg/ccgtwcl6fdtgl2374kpssn4xdc2c7daiyqxkvwodvnrlvynrurq6.py
# Topologically Sorted Source Nodes: [result_1, m, ge, n, x, index_1, ge_1, result_2, batched_outputs_2, mask_2, mask_3, mask_block_sum, gt, lt, partial_blocks, partial_blocks_1, dense_mask, full_blocks, full_blocks_1, dense_mask_1], Original ATen: [aten.view, aten.arange, aten.ge, aten.bitwise_and, aten.index, aten.unsqueeze, aten.permute, aten.sum, aten.gt, aten.lt, aten._to_copy, aten.eq]
# Source node to ATen node mapping:
#   batched_outputs_2 => unsqueeze
#   dense_mask => convert_element_type_2
#   dense_mask_1 => convert_element_type_5
#   full_blocks => eq
#   full_blocks_1 => convert_element_type_1
#   ge => ge, view
#   ge_1 => ge_1, view_2
#   gt => gt
#   index_1 => index_1
#   lt => lt
#   m => iota_2
#   mask_2 => view_4
#   mask_3 => permute
#   mask_block_sum => sum_1
#   n => iota_3
#   partial_blocks => bitwise_and_2
#   partial_blocks_1 => convert_element_type
#   result_1 => bitwise_and, full_default
#   result_2 => bitwise_and_1
#   x => index
# Graph fragment:
#   %arg0_1 : Tensor "i32[2048][1]cuda:2" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "i32[2176][1]cuda:2" = PlaceHolder[target=arg1_1]
#   %sum_1 : Tensor "i64[1, 1, 16, 16][256, 256, 16, 1]cuda:2" = PlaceHolder[target=sum_1]
#   %full_default : Tensor "b8[1, 1, 1][1, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %iota_2 : Tensor "i64[2048][1]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (2048,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:2, requires_grad: False})
#   %view : Tensor "i64[2048, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [2048, 1]), kwargs = {})
#   %iota_3 : Tensor "i64[2048][1]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (2048,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:2, requires_grad: False})
#   %ge : Tensor "b8[2048, 2048][2048, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%view, %iota_3), kwargs = {})
#   %bitwise_and : Tensor "b8[1, 2048, 2048][4194304, 2048, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %ge), kwargs = {})
#   %index : Tensor "i32[2048][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%iota_2]), kwargs = {})
#   %index_1 : Tensor "i32[2048][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%index]), kwargs = {})
#   %view_2 : Tensor "i32[2048, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [2048, 1]), kwargs = {})
#   %ge_1 : Tensor "b8[2048, 2048][2048, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%iota_3, %view_2), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 2048, 2048][4194304, 2048, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %ge_1), kwargs = {})
#   %unsqueeze : Tensor "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%bitwise_and_1, 1), kwargs = {})
#   %view_4 : Tensor "b8[1, 1, 16, 128, 16, 128][4194304, 4194304, 262144, 2048, 128, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand, [1, 1, 16, 128, 16, 128]), kwargs = {})
#   %permute : Tensor "b8[1, 1, 16, 16, 128, 128][4194304, 4194304, 262144, 128, 2048, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 1, 2, 4, 3, 5]), kwargs = {})
#   %sum_1 : Tensor "i64[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute, [-2, -1]), kwargs = {})
#   %gt : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_1, 0), kwargs = {})
#   %lt : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt, %lt), kwargs = {})
#   %convert_element_type : Tensor "i8[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_2, torch.int8), kwargs = {})
#   %convert_element_type_2 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.int32), kwargs = {})
#   %eq : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %convert_element_type_1 : Tensor "i8[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.int8), kwargs = {})
#   %convert_element_type_5 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.int32), kwargs = {})
#   return %sum_1,%convert_element_type_2,%convert_element_type_5
triton_red_fused__to_copy_arange_bitwise_and_eq_ge_gt_index_lt_permute_sum_unsqueeze_view_0 = async_compile.triton('triton_red_fused__to_copy_arange_bitwise_and_eq_ge_gt_index_lt_permute_sum_unsqueeze_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 16384},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_arange_bitwise_and_eq_ge_gt_index_lt_permute_sum_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 4096, 'r0_': 0}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_arange_bitwise_and_eq_ge_gt_index_lt_permute_sum_unsqueeze_view_0(in_ptr0, in_ptr1, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 16
    x0 = (xindex % 16)
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 128
        r0_2 = (r0_index % 128)
        tmp7 = tl.load(in_ptr0 + (r0_3 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = (r0_3 + 128*x1).to(tl.int64)
        tmp1 = (tmp0).to(tl.int64)
        tmp2 = (r0_2 + 128*x0).to(tl.int64)
        tmp3 = (tmp2).to(tl.int64)
        tmp4 = tmp1 >= tmp3
        tmp5 = tl.full([1, 1], True, tl.int1)
        tmp6 = tmp5 & tmp4
        tmp8 = (tl.full([1, 1], 2176, tl.int32)).to(tl.int32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp7 < 0
        tmp11 = tl.where(tmp10, tmp9, tmp7)
        tl.device_assert(((0 <= tmp11) & (tmp11 < 2176)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp11 < 2176")
        tmp13 = tl.load(in_ptr1 + (tmp11), r0_mask & xmask, eviction_policy='evict_last')
        tmp14 = tmp13.to(tl.int64)
        tmp15 = tmp3 >= tmp14
        tmp16 = tmp6 & tmp15
        tmp17 = tmp16.to(tl.int64)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.full([1, 1], 0, tl.int64)
    tmp22 = tmp19 > tmp21
    tmp23 = tl.full([1, 1], 16384, tl.int64)
    tmp24 = tmp19 < tmp23
    tmp25 = tmp22 & tmp24
    tmp26 = tmp25.to(tl.int8)
    tmp27 = tmp26.to(tl.int32)
    tmp28 = tmp19 == tmp23
    tmp29 = tmp28.to(tl.int8)
    tmp30 = tmp29.to(tl.int32)
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp30, xmask)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/4v/c4vxsei4syuvddr7savecnokl7jkbml5meirvg6mfitfp6ipeh27.py
# Topologically Sorted Source Nodes: [col_range, num_blocks_in_row, child_3, unsqueeze_1, index_mask, child_4, valid_indices], Original ATen: [aten.arange, aten.sum, aten._to_copy, aten.unsqueeze, aten.lt, aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   child_3 => convert_element_type_3
#   child_4 => convert_element_type_4
#   col_range => iota_5
#   index_mask => lt_1
#   num_blocks_in_row => sum_2
#   unsqueeze_1 => unsqueeze_2
#   valid_indices => full_default_2, where
# Graph fragment:
#   %convert_element_type_2 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:2" = PlaceHolder[target=convert_element_type_2]
#   %sum_2 : Tensor "i64[1, 1, 16][16, 16, 1]cuda:2" = PlaceHolder[target=sum_2]
#   %getitem_1 : Tensor "i64[1, 1, 16, 16][256, 256, 16, 1]cuda:2" = PlaceHolder[target=getitem_1]
#   %convert_element_type_3 : Tensor "i32[1, 1, 16][16, 16, 1]cuda:2" = PlaceHolder[target=convert_element_type_3]
#   %convert_element_type_4 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:2" = PlaceHolder[target=convert_element_type_4]
#   %iota_5 : Tensor "i32[16][1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:2, requires_grad: False})
#   %sum_2 : Tensor "i64[1, 1, 16][16, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_2, [-1]), kwargs = {})
#   %convert_element_type_3 : Tensor "i32[1, 1, 16][16, 16, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.int32), kwargs = {})
#   %unsqueeze_2 : Tensor "i32[1, 1, 16, 1][16, 16, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 3), kwargs = {})
#   %lt_1 : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_5, %unsqueeze_2), kwargs = {})
#   %convert_element_type_4 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1, torch.int32), kwargs = {})
#   %full_default_2 : Tensor "i32[][]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 16), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %where : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_1, %convert_element_type_4, %full_default_2), kwargs = {})
#   return %sum_2,%convert_element_type_3,%convert_element_type_4,%where
triton_per_fused__to_copy_arange_lt_scalar_tensor_sum_unsqueeze_where_1 = async_compile.triton('triton_per_fused__to_copy_arange_lt_scalar_tensor_sum_unsqueeze_where_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*i64', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'out_ptr3': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy_arange_lt_scalar_tensor_sum_unsqueeze_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 128, 'r0_': 7168}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__to_copy_arange_lt_scalar_tensor_sum_unsqueeze_where_1(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp7 = tl.load(in_ptr1 + (r0_1 + 16*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.int64)
    tmp6 = tmp5.to(tl.int32)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = (r0_1).to(tl.int32)
    tmp10 = tmp9 < tmp6
    tmp11 = tl.full([1, 1], 16, tl.int32)
    tmp12 = tl.where(tmp10, tmp8, tmp11)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr2 + (r0_1 + 16*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_1 + 16*x0), tmp12, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/63/c63zeel2miqm3jsajoidkmpx3k3sw5kp3r7czqu42xegfcvvf7wj.py
# Topologically Sorted Source Nodes: [dense_mask_2], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   dense_mask_2 => full_default_1
# Graph fragment:
#   %full_default_1 : Tensor "i32[1, 1, 16, 17][272, 272, 17, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 16, 17], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   return %index_put
triton_poi_fused_new_zeros_2 = async_compile.triton('triton_poi_fused_new_zeros_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_new_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 2176}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/fd/cfdwpmtijuqhwcjtcrp2qqjlkdpqacbhuhvur2bclcmphvq773ix.py
# Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   setitem => full_default_3
# Graph fragment:
#   %full_default_3 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1, 1], 1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:2, pin_memory: False})
#   return %full_default_3
triton_poi_fused_view_3 = async_compile.triton('triton_poi_fused_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.full([1], 1, tl.int32)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/v5/cv553zp42suh6d62odrrlbrhnjipgf6bztu3rrjfcvnlj5syznln.py
# Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   setitem => iota_7
# Graph fragment:
#   %iota_7 : Tensor "i64[1][1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:2, requires_grad: False})
#   return %iota_7
triton_poi_fused_arange_4 = async_compile.triton('triton_poi_fused_arange_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/pa/cparlcp2hubgu23slpuhgh5t6qashedzo3sga3zbb5ufmog5du7d.py
# Topologically Sorted Source Nodes: [arange_4], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   arange_4 => iota_4
# Graph fragment:
#   %iota_4 : Tensor "i32[16][1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:2, requires_grad: False})
#   return %iota_4
triton_poi_fused_arange_5 = async_compile.triton('triton_poi_fused_arange_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 128}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = (x0).to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/eu/ceuqjkxbziqfawyv4ergfrldo2dur7xlw67cxictjhidulog7ojf.py
# Topologically Sorted Source Nodes: [batched_outputs_3, transpose, num_blocks_in_row_2, q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
# Source node to ATen node mapping:
#   batched_outputs_3 => clone_4, slice_2
#   num_blocks_in_row_2 => sum_4
#   q_num_blocks => convert_element_type_8
#   transpose => permute_1
# Graph fragment:
#   %buf18 : Tensor  = PlaceHolder[target=buf18]
#   %clone_4 : Tensor "i32[1, 1, 16, 16][256, 1, 16, 1]cuda:2" = PlaceHolder[target=clone_4]
#   %sum_4 : Tensor "i64[1, 1, 16][16, 16, 1]cuda:2" = PlaceHolder[target=sum_4]
#   %slice_2 : Tensor "i32[1, 1, 16, 16][272, 272, 17, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%index_put, 3, 0, 16), kwargs = {})
#   %clone_4 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_1 : Tensor "i32[1, 1, 16, 16][256, 256, 1, 16]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%clone_4, [0, 1, 3, 2]), kwargs = {})
#   %sum_4 : Tensor "i64[1, 1, 16][16, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_1, [-1]), kwargs = {})
#   %convert_element_type_8 : Tensor "i32[1, 1, 16][16, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_4, torch.int32), kwargs = {})
#   return %clone_4,%sum_4,%convert_element_type_8
triton_per_fused__to_copy_clone_slice_sum_transpose_6 = async_compile.triton('triton_per_fused__to_copy_clone_slice_sum_transpose_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr2': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy_clone_slice_sum_transpose_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 2176, 'r0_': 0}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__to_copy_clone_slice_sum_transpose_6(in_ptr0, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 17*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.int64)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0 + 16*r0_1), tmp0, r0_mask & xmask)
    tl.store(out_ptr2 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/2r/c2rigni4jlryy7fd7i3efxwjucflsimuujltopg2qjmvumf24rff.py
# Topologically Sorted Source Nodes: [full_q_indices], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   full_q_indices => clone_9, convert_element_type_11
# Graph fragment:
#   %getitem_7 : Tensor "i64[1, 1, 16, 16][256, 256, 1, 16]cuda:2" = PlaceHolder[target=getitem_7]
#   %convert_element_type_11 : Tensor "i32[1, 1, 16, 16][256, 256, 1, 16]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_7, torch.int32), kwargs = {})
#   %clone_9 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_11,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_9
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'y': 2048, 'x': 2048}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x1), xmask & ymask)
    tmp1 = tmp0.to(tl.int32)
    tl.store(out_ptr0 + (x1 + 16*y0), tmp1, xmask & ymask)
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
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride_grouped((arg0_1, arg1_1), ((2048, ), (2176, )), ((1, ), (1, )), 'input')
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_cuda((1, 1, 16, 16), (256, 256, 16, 1), torch.int32)
            buf5 = empty_strided_cuda((1, 1, 16, 16), (256, 256, 16, 1), torch.int32)
            # Topologically Sorted Source Nodes: [result_1, m, ge, n, x, index_1, ge_1, result_2, batched_outputs_2, mask_2, mask_3, mask_block_sum, gt, lt, partial_blocks, partial_blocks_1, dense_mask, full_blocks, full_blocks_1, dense_mask_1], Original ATen: [aten.view, aten.arange, aten.ge, aten.bitwise_and, aten.index, aten.unsqueeze, aten.permute, aten.sum, aten.gt, aten.lt, aten._to_copy, aten.eq]
            raw_stream2 = get_raw_stream(2)
            triton_red_fused__to_copy_arange_bitwise_and_eq_ge_gt_index_lt_permute_sum_unsqueeze_view_0.run(arg0_1, arg1_1, buf1, buf5, 256, 16384, stream=raw_stream2)
            del arg0_1
            del arg1_1
            # Topologically Sorted Source Nodes: [gt, lt, partial_blocks, partial_blocks_1, dense_mask, col_indices], Original ATen: [aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.sort]
            buf2 = torch.ops.aten.sort.stable(buf1, stable=True, descending=True)
            buf4 = buf2[1]
            assert_tensor_metadata(buf4, (1, 1, 16, 16), (256, 256, 16, 1), torch.int64, 'torch.ops.aten.sort.stable')
            assert_alignment(buf4, 16, 'torch.ops.aten.sort.stable')
            del buf2
            # Topologically Sorted Source Nodes: [full_blocks, full_blocks_1, dense_mask_1, col_indices_1], Original ATen: [aten.eq, aten._to_copy, aten.sort]
            buf6 = torch.ops.aten.sort.stable(buf5, stable=True, descending=True)
            buf8 = buf6[1]
            assert_tensor_metadata(buf8, (1, 1, 16, 16), (256, 256, 16, 1), torch.int64, 'torch.ops.aten.sort.stable')
            assert_alignment(buf8, 16, 'torch.ops.aten.sort.stable')
            del buf6
            buf10 = empty_strided_cuda((1, 1, 16), (16, 16, 1), torch.int32)
            buf11 = empty_strided_cuda((1, 1, 16, 16), (256, 256, 16, 1), torch.int32)
            buf17 = empty_strided_cuda((1, 1, 16, 16), (256, 256, 16, 1), torch.int32)
            # Topologically Sorted Source Nodes: [col_range, num_blocks_in_row, child_3, unsqueeze_1, index_mask, child_4, valid_indices], Original ATen: [aten.arange, aten.sum, aten._to_copy, aten.unsqueeze, aten.lt, aten.scalar_tensor, aten.where]
            raw_stream2 = get_raw_stream(2)
            triton_per_fused__to_copy_arange_lt_scalar_tensor_sum_unsqueeze_where_1.run(buf1, buf4, buf10, buf11, buf17, 16, 16, stream=raw_stream2)
            del buf4
            buf12 = empty_strided_cuda((1, 1, 16, 17), (272, 272, 17, 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_2], Original ATen: [aten.new_zeros]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_new_zeros_2.run(buf12, 272, stream=raw_stream2)
            buf13 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
            # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.view]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_view_3.run(buf13, 1, stream=raw_stream2)
            buf14 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.arange]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_arange_4.run(buf14, 1, stream=raw_stream2)
            buf15 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.arange]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_arange_4.run(buf15, 1, stream=raw_stream2)
            buf16 = empty_strided_cuda((16, ), (1, ), torch.int32)
            # Topologically Sorted Source Nodes: [arange_4], Original ATen: [aten.arange]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_arange_5.run(buf16, 16, stream=raw_stream2)
            aten.index_put_(buf12, [reinterpret_tensor(buf14, (1, 1, 1, 1), (0, 0, 0, 0), 0), reinterpret_tensor(buf15, (1, 1, 1), (0, 0, 0), 0), reinterpret_tensor(buf16, (16, 1), (1, 0), 0), buf17], buf13, False)
            del buf13
            del buf14
            del buf15
            buf19 = reinterpret_tensor(buf17, (1, 1, 16, 16), (256, 1, 16, 1), 0); del buf17  # reuse
            buf42 = reinterpret_tensor(buf16, (1, 1, 16), (16, 16, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [batched_outputs_3, transpose, num_blocks_in_row_2, q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
            raw_stream2 = get_raw_stream(2)
            triton_per_fused__to_copy_clone_slice_sum_transpose_6.run(buf12, buf19, buf42, 16, 16, stream=raw_stream2)
            del buf12
            # Topologically Sorted Source Nodes: [batched_outputs_3, transpose, col_indices_2], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort]
            buf20 = torch.ops.aten.sort.stable(reinterpret_tensor(buf19, (1, 1, 16, 16), (0, 0, 1, 16), 0), stable=True, descending=True)
            buf22 = buf20[1]
            assert_tensor_metadata(buf22, (1, 1, 16, 16), (256, 256, 1, 16), torch.int64, 'torch.ops.aten.sort.stable')
            assert_alignment(buf22, 16, 'torch.ops.aten.sort.stable')
            del buf20
            buf24 = empty_strided_cuda((1, 1, 16), (16, 16, 1), torch.int32)
            buf25 = reinterpret_tensor(buf19, (1, 1, 16, 16), (256, 256, 16, 1), 0); del buf19  # reuse
            buf31 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [col_range_1, num_blocks_in_row_1, child_7, unsqueeze_3, index_mask_1, child_8, valid_indices_1], Original ATen: [aten.arange, aten.sum, aten._to_copy, aten.unsqueeze, aten.lt, aten.scalar_tensor, aten.where]
            raw_stream2 = get_raw_stream(2)
            triton_per_fused__to_copy_arange_lt_scalar_tensor_sum_unsqueeze_where_1.run(buf5, buf8, buf24, buf25, buf31, 16, 16, stream=raw_stream2)
            del buf8
            buf26 = empty_strided_cuda((1, 1, 16, 17), (272, 272, 17, 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_4], Original ATen: [aten.new_zeros]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_new_zeros_2.run(buf26, 272, stream=raw_stream2)
            buf27 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
            # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.view]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_view_3.run(buf27, 1, stream=raw_stream2)
            buf28 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.arange]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_arange_4.run(buf28, 1, stream=raw_stream2)
            buf29 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.arange]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_arange_4.run(buf29, 1, stream=raw_stream2)
            buf30 = empty_strided_cuda((16, ), (1, ), torch.int32)
            # Topologically Sorted Source Nodes: [arange_6], Original ATen: [aten.arange]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused_arange_5.run(buf30, 16, stream=raw_stream2)
            aten.index_put_(buf26, [reinterpret_tensor(buf28, (1, 1, 1, 1), (0, 0, 0, 0), 0), reinterpret_tensor(buf29, (1, 1, 1), (0, 0, 0), 0), reinterpret_tensor(buf30, (16, 1), (1, 0), 0), buf31], buf27, False)
            del buf27
            del buf28
            del buf29
            buf33 = reinterpret_tensor(buf31, (1, 1, 16, 16), (256, 1, 16, 1), 0); del buf31  # reuse
            buf39 = reinterpret_tensor(buf30, (1, 1, 16), (16, 16, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [batched_outputs_5, transpose_1, num_blocks_in_row_3, full_q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
            raw_stream2 = get_raw_stream(2)
            triton_per_fused__to_copy_clone_slice_sum_transpose_6.run(buf26, buf33, buf39, 16, 16, stream=raw_stream2)
            del buf26
            # Topologically Sorted Source Nodes: [batched_outputs_5, transpose_1, col_indices_3], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort]
            buf34 = torch.ops.aten.sort.stable(reinterpret_tensor(buf33, (1, 1, 16, 16), (0, 0, 1, 16), 0), stable=True, descending=True)
            buf36 = buf34[1]
            assert_tensor_metadata(buf36, (1, 1, 16, 16), (256, 256, 1, 16), torch.int64, 'torch.ops.aten.sort.stable')
            assert_alignment(buf36, 16, 'torch.ops.aten.sort.stable')
            del buf34
            buf37 = reinterpret_tensor(buf33, (1, 1, 16, 16), (256, 256, 16, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [full_q_indices], Original ATen: [aten._to_copy]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy_7.run(buf36, buf37, 16, 16, stream=raw_stream2)
            del buf36
            buf40 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [q_indices], Original ATen: [aten._to_copy]
            raw_stream2 = get_raw_stream(2)
            triton_poi_fused__to_copy_7.run(buf22, buf40, 16, 16, stream=raw_stream2)
            del buf22
        return (buf37, buf39, buf40, buf42, buf25, buf24, buf11, buf10, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, ), (1, ), device='cuda:2', dtype=torch.int32)
    arg1_1 = rand_strided((2176, ), (1, ), device='cuda:2', dtype=torch.int32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
