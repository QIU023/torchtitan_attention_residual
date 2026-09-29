
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*i32', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
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
