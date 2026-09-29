
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
