
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_flex_attention_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 197120}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_flex_attention_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x4 = xindex // 128
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 4)
    x2 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (128 + x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (16384 + x3), None)
    tmp31 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp3 = tmp1 > tmp2
    tmp4 = tmp1 == tmp2
    tmp5 = tmp1 != tmp1
    tmp6 = tmp2 != tmp2
    tmp7 = tmp5 > tmp6
    tmp8 = tmp3 | tmp7
    tmp9 = tmp5 & tmp6
    tmp10 = tmp4 | tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = tmp10 & tmp13
    tmp15 = tmp8 | tmp14
    tmp16 = tl.where(tmp15, tmp1, tmp2)
    tmp17 = tl.where(tmp15, tmp11, tmp12)
    tmp18 = tl.full([1], float("-inf"), tl.float32)
    tmp19 = tmp16 == tmp18
    tmp20 = tmp1 - tmp16
    tmp21 = tl.full([1], 0.0, tl.float32)
    tmp22 = tl.where(tmp19, tmp21, tmp20)
    tmp23 = libdevice.exp2(tmp22)
    tmp24 = tmp0 * tmp23
    tmp26 = tmp2 - tmp16
    tmp27 = tl.where(tmp19, tmp21, tmp26)
    tmp28 = libdevice.exp2(tmp27)
    tmp29 = tmp25 * tmp28
    tmp30 = tmp24 + tmp29
    tmp32 = (tmp30 / tmp31)
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 128*x2 + 4096*x1), tmp33, None)
