
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
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*i64', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'out_ptr3': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
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
