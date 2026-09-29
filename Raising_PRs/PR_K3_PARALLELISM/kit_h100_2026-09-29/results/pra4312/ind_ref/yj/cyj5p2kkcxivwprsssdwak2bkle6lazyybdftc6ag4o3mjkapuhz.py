
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i64', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_arange_constant_pad_nd_eq_index_permute_sum_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 8, 'r0_': 32}, 'backend_hash': '50B62BD4A10FB8E3194387AC3394035BF7203341178C85B8EE414B155096DFE5', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused_arange_constant_pad_nd_eq_index_permute_sum_unsqueeze_view_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    x1 = xindex // 2
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    x2 = xindex // ks2
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = (x1 + (r0_3 // 128)*(triton_helpers.div_floor_integer(127 + ks0,  128)) + 64*x0*(triton_helpers.div_floor_integer(127 + ks0,  128))).to(tl.int32)
        tmp1 = (ks1).to(tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = ((r0_3 % 128)).to(tl.int32)
        tmp4 = (ks0).to(tl.int32)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp2 & tmp5
        tl.device_assert((x1 + (r0_3 // 128)*(triton_helpers.div_floor_integer(127 + ks0,  128)) + 64*x0*(triton_helpers.div_floor_integer(127 + ks0,  128)) < 4) | ~(r0_mask & tmp6 & xmask), "index out of bounds: x1 + (r0_3 // 128)*(triton_helpers.div_floor_integer(127 + ks0,  128)) + 64*x0*(triton_helpers.div_floor_integer(127 + ks0,  128)) < 4")
        tmp8 = tl.load(in_ptr0 + (x1 + (r0_3 // 128)*(triton_helpers.div_floor_integer(127 + ks0,  128)) + 64*x0*(triton_helpers.div_floor_integer(127 + ks0,  128))), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tl.device_assert((tl.broadcast_to((r0_3 % 128), [XBLOCK, R0_BLOCK]) < 4) | ~(r0_mask & tmp6 & xmask), "index out of bounds: tl.broadcast_to((r0_3 % 128), [XBLOCK, R0_BLOCK]) < 4")
        tmp10 = tl.load(in_ptr0 + (tl.broadcast_to((r0_3 % 128), [XBLOCK, R0_BLOCK])), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp8 == tmp10
        tmp12 = tl.full(tmp11.shape, False, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp13.to(tl.int64)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 2*x1 + 2*x2*(triton_helpers.div_floor_integer(127 + ks0,  128))), tmp16, xmask)
