# AOT ID: ['1_backward']
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


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/jz/cjzcga4cpg6zfz54rza4rugp53dya7ectay6gs7kg2bxxdep7vzs.py
# Topologically Sorted Source Nodes: [flex_attention_backward], Original ATen: [flex_attention_backward]
# Source node to ATen node mapping:
#   flex_attention_backward => flex_attention_backward
# Graph fragment:
#   %getitem : Tensor "bf16[1, 32, 2048, 64][4194304, 64, 2048, 1]cuda:1" = PlaceHolder[target=getitem]
#   %tangents_1 : Tensor "bf16[1, 32, 2048, 64][4194304, 64, 2048, 1]cuda:1" = PlaceHolder[target=tangents_1]
#   %buf0 : Tensor "bf16[1, 32, 2048][65536, 1, 32]cuda:1" = PlaceHolder[target=buf0]
#   %flex_attention_backward : [num_users=3] = call_function[target=torch.ops.higher_order.flex_attention_backward](args = (%primals_1, %primals_2, %primals_3, %getitem, %getitem_1, %tangents_1, None, %fw_graph0, %joint_graph0, (2048, 2048, %primals_5, %primals_4, %primals_8, %primals_9, %primals_10, %primals_11, %primals_12, %primals_13, None, None, None, None, 128, 128, %mask_graph0), 0.10206207261596575, {BACKEND: AUTO, PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_6, %primals_7)), kwargs = {})
#   return %buf0,%buf1
triton_per_fused_flex_attention_backward_0 = async_compile.triton('triton_per_fused_flex_attention_backward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'y': 2048, 'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_per_fused_flex_attention_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'y': 524288, 'x': 0, 'r0_': 16777216}, 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused_flex_attention_backward_0(in_ptr0, in_ptr1, out_ptr1, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 32
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x1 + 2048*y0), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 64*x1 + 2048*y0), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 2)[:, :, None].to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 2048*x1), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/bound/cache_pp4vp4/ic/3n/c3ns5gsdtuprheiqju54gu7zphjdpe3ca4e5fskzuejdiy4q52ry.py
# Topologically Sorted Source Nodes: [flex_attention_backward], Original ATen: [flex_attention_backward]
# Source node to ATen node mapping:
#   flex_attention_backward => flex_attention_backward
# Graph fragment:
#   %primals_1 : Tensor "bf16[1, 32, 2048, 96][3072, 96, 3072, 1]cuda:1" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "bf16[1, 32, 2048, 96][3072, 96, 3072, 1]cuda:1" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[1, 32, 2048, 64][4096, 128, 4096, 1]cuda:1" = PlaceHolder[target=primals_3]
#   %getitem_1 : Tensor "f32[1, 32, 2048][65536, 2048, 1]cuda:1" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[1, 32, 2048][65536, 2048, 1]cuda:1" = PlaceHolder[target=buf1]
#   %tangents_1 : Tensor "bf16[1, 32, 2048, 64][4194304, 64, 2048, 1]cuda:1" = PlaceHolder[target=tangents_1]
#   %getitem_3 : Tensor "bf16[1, 32, 2048, 96][6291456, 96, 3072, 1]cuda:1" = PlaceHolder[target=getitem_3]
#   %getitem_5 : Tensor "bf16[1, 32, 2048, 64][4194304, 64, 2048, 1]cuda:1" = PlaceHolder[target=getitem_5]
#   %primals_5 : Tensor "i32[1, 1, 16][16, 16, 1]cuda:1" = PlaceHolder[target=primals_5]
#   %primals_4 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:1" = PlaceHolder[target=primals_4]
#   %primals_10 : Tensor "i32[1, 1, 16][16, 16, 1]cuda:1" = PlaceHolder[target=primals_10]
#   %primals_11 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:1" = PlaceHolder[target=primals_11]
#   %primals_8 : Tensor "i32[1, 1, 16][16, 16, 1]cuda:1" = PlaceHolder[target=primals_8]
#   %primals_9 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:1" = PlaceHolder[target=primals_9]
#   %primals_12 : Tensor "i32[1, 1, 16][16, 16, 1]cuda:1" = PlaceHolder[target=primals_12]
#   %primals_13 : Tensor "i32[1, 1, 16, 16][256, 256, 16, 1]cuda:1" = PlaceHolder[target=primals_13]
#   %primals_6 : Tensor "i32[2048][1]cuda:1" = PlaceHolder[target=primals_6]
#   %primals_7 : Tensor "i32[2176][1]cuda:1" = PlaceHolder[target=primals_7]
#   %flex_attention_backward : [num_users=3] = call_function[target=torch.ops.higher_order.flex_attention_backward](args = (%primals_1, %primals_2, %primals_3, %getitem, %getitem_1, %tangents_1, None, %fw_graph0, %joint_graph0, (2048, 2048, %primals_5, %primals_4, %primals_8, %primals_9, %primals_10, %primals_11, %primals_12, %primals_13, None, None, None, None, 128, 128, %mask_graph0), 0.10206207261596575, {BACKEND: AUTO, PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_6, %primals_7)), kwargs = {})
#   return %getitem_4
triton_tem_fused_flex_attention_backward_1 = async_compile.triton('triton_tem_fused_flex_attention_backward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=4,
triton_meta={'signature': {'arg_Q': '*bf16', 'arg_K': '*bf16', 'arg_V': '*bf16', 'arg_LSE': '*fp32', 'arg_DELTA': '*fp32', 'arg_DO': '*bf16', 'arg_DQ': '*bf16', 'arg_DV': '*bf16', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_Q_NUM_BLKS': '*i32', 'arg_Q_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'arg_FULL_Q_NUM_BLKS': '*i32', 'arg_FULL_Q_IDX': '*i32', 'in_ptr16': '*i32', 'in_ptr17': '*i32', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_flex_attention_backward_1', 'backend_hash': '877E62F80237AD7512133A78967DC55708298B1453FFCC61EFCD3C7B3600FFCC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False, 'FLOAT32_PRECISION': "'tf32'", 'IS_DIVISIBLE': True, 'SM_SCALE': 0.10206207261596575, 'GQA_SHARED_HEADS': 1, 'HAS_FULL_BLOCKS': True, 'QK_HEAD_DIM': 96, 'QK_HEAD_DIM_ROUNDED': 128, 'V_HEAD_DIM': 64, 'V_HEAD_DIM_ROUNDED': 64, 'SAFE_HEAD_DIM': False, 'USE_TMA': False, 'BLOCK_M1': 64, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 64, 'SPARSE_Q_BLOCK_SIZE': 128, 'SPARSE_KV_BLOCK_SIZE': 128}},

)
@triton.jit
def triton_tem_fused_flex_attention_backward_1(arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0):
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.10206207261596575
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 96
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = False
    USE_TMA : tl.constexpr = False
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    DELTA = arg_DELTA
    DO = arg_DO
    DQ = arg_DQ
    DV = arg_DV
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    Q_NUM_BLKS = arg_Q_NUM_BLKS
    Q_IDX = arg_Q_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX
    FULL_Q_NUM_BLKS = arg_FULL_Q_NUM_BLKS
    FULL_Q_IDX = arg_FULL_Q_IDX

    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT*DO, axis=-1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key, is the written to via the store_output call due to some limitations with
    # inductor codegen
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries or keys/values, d: Head dim
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    # (Modifiable) Performance tuning options
    # BLOCK_M1: when calculating DK & DV, iterate over BLOCK_M1 across the seqlen dim of Q in each thread block.
    # BLOCK_N1: when calculating DK & DV, the thread block size across the seqlen dim of K/V.
    # BLOCK_M2: when calculating DQ, the thread block size across the seqlen dim of Q.
    # BLOCK_N2: when calculating DQ, iterate over BLOCK_N2 across the seqlen dim of K/V in each thread block.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # Q_NUM_BLKS: The number of Q blocks (that may or may not require masking) for each query.
    # Q_IDX: The indices of Q blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_Q_NUM_BLKS: The number of fully unmasked Q blocks (so we don't need masking) for each query.
    # FULL_Q_IDX: The indices of fully unmasked Q blocks (so we don't need masking) for each query.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qd = 3072, 96, 3072, 1
    stride_kz, stride_kh, stride_kn, stride_kd = 3072, 96, 3072, 1
    stride_vz, stride_vh, stride_vn, stride_vd = 4096, 128, 4096, 1
    stride_doz, stride_doh, stride_dom, stride_dod = 4194304, 64, 2048, 1

    stride_dqz, stride_dqh, stride_dqm, stride_dqd = 6291456, 96, 3072, 1
    stride_dvz, stride_dvh, stride_dvm, stride_dvd = 4194304, 64, 2048, 1

    ZQ = 1
    HQ = 32
    HKV = 32
    Q_LEN = 2048
    ZKV = 1
    KV_LEN = 2048

    MATMUL_PRECISION = Q.dtype.element_ty

    pid = tl.program_id(0).to(INDEX_DTYPE)
    NUM_KV_BLOCKS = tl.cdiv(KV_LEN, BLOCK_N1)
    NUM_Q_BLOCKS = tl.cdiv(Q_LEN, BLOCK_M2)

    off_zq = tl.program_id(1).to(INDEX_DTYPE) # q batch idx
    off_hkv = tl.program_id(2).to(INDEX_DTYPE) # kv head idx
    off_zkv = off_zq % ZKV # kv batch idx

    SPARSE_Z = 1
    SPARSE_HQ = 1

    sparse_idx_z = off_zq % SPARSE_Z

    k_adj = (stride_kh * off_hkv + stride_kz * off_zkv).to(tl.int64)
    v_adj = (stride_vh * off_hkv + stride_vz * off_zkv).to(tl.int64)
    # first compute broadcasted dv of shape [Bq, Hkv, KV_LEN, V_HEAD_DIM]
    # then reduce to dv of shape [Bkv, Hkv, KV_LEN, V_HEAD_DIM]
    dv_adj = (stride_dvh * off_hkv + stride_dvz * off_zq).to(tl.int64)

    # offset K, V, DV pointers for batch/kv-head
    K += k_adj
    V += v_adj
    DV += dv_adj

    RCP_LN2 = 1.44269504
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    if pid >= NUM_KV_BLOCKS:
        off_pid = pid - NUM_KV_BLOCKS
        # THIS BLOCK DOES DQ
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M2)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
        off_hq2 = off_pid // NUM_Q_BLOCKS + off_hkv * GQA_SHARED_HEADS
        start_m2_block = off_pid % NUM_Q_BLOCKS
        off_pid_mask = start_m2_block // SPARSE_Q_MULTIPLE
        stride_kv_num_blks_h = 16
        stride_kv_idx_h = 256
        stride_kv_idx_m = 16
        stride_full_kv_num_blks_h = 16
        stride_full_kv_idx_h = 256
        stride_full_kv_idx_m = 16

        sparse_idx_hq2 = off_hq2 % SPARSE_HQ
        sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq2

        sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + off_pid_mask
        sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + off_pid_mask * stride_kv_idx_m  # noqa: B950
        sparse_full_kv_num_blks_offset = sparse_hz_offset * stride_full_kv_num_blks_h + off_pid_mask
        sparse_full_kv_idx_offset = sparse_hz_offset * stride_full_kv_idx_h + off_pid_mask * stride_full_kv_idx_m  # noqa: B950

        # Offset Q, DQ, DO, DELTA & LSE. These inputs are offsetted by query heads.
        q_adj2 = (stride_qh * off_hq2 + stride_qz * off_zq).to(tl.int64)
        do_adj2 = (stride_doh * off_hq2 + stride_doz * off_zq).to(tl.int64)
        dq_adj2 = (stride_dqh * off_hq2 + stride_dqz * off_zq).to(tl.int64)
        off_chz2 = ((off_zq * HQ + off_hq2) * Q_LEN).to(tl.int64)

        Q2 = Q + q_adj2
        DO2 = DO + do_adj2
        # TODO: This does not work if DQ is not the same layout as Q (for example,
        # if Q is broadcasted)
        DQ2 = DQ + dq_adj2
        LSE2 = LSE + off_chz2
        DELTA2 = DELTA + off_chz2

        dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        start_m2 = start_m2_block * BLOCK_M2
        offs_m2 = start_m2 + tl.arange(0, BLOCK_M2)

        desc_q = None
        desc_do = None
        desc_k_dq = None
        desc_v_dq = None
        # load Q and do: they stay in SRAM throughout the inner loop.
        q = load_checked_2d(Q2, offs_m2, offs_k, stride_qm, stride_qd, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, QK_HEAD_DIM, INDEX_DTYPE)
        do = load_checked_2d(DO2, offs_m2, offs_v, stride_dom, stride_dod, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM, INDEX_DTYPE)
        if PRESCALE_QK:
            q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        if IS_DIVISIBLE:
            Di = tl.load(DELTA2 + offs_m2)
            lse = tl.load(LSE2 + offs_m2)
        else:
            Di = tl.load(DELTA2 + offs_m2, mask=offs_m2 < Q_LEN)
            lse = tl.load(LSE2 + offs_m2, mask=offs_m2 < Q_LEN)
        lse = tl.where(lse == -float("inf"), 0.0, lse)
        lse = lse[:, None]

        # ~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # KV_IDX and KV_NUM_BLKS are always contiguous.
        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        sparse_kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        dq = bwd_dq_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
            K, V, desc_k_dq, desc_v_dq, kv_start, start_m2,
            dq, q, do, Di, lse,
            off_zq, off_hq2,
            stride_kn, stride_kd, stride_vn, stride_vd,
            kv_indices, sparse_kv_num_blocks,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=False,
        )

        if HAS_FULL_BLOCKS:
            # ~~~~~~~~~~~ partial unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
            kv_indices = FULL_KV_IDX + sparse_full_kv_idx_offset
            kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
            sparse_kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_full_kv_num_blks_offset)

            offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
            dq = bwd_dq_inner(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
                K, V, desc_k_dq, desc_v_dq, kv_start, start_m2,
                dq, q, do, Di, lse,
                off_zq, off_hq2,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=True,
            )

        # Write back dQ.
        dq_ptrs = DQ2 + offs_m2[:, None].to(INDEX_DTYPE) * stride_dqm + offs_k[None, :].to(INDEX_DTYPE) * stride_dqd
        dq *= SM_SCALE
        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=(offs_m2[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM))
    else:
        # THIS BLOCK DOES DK & DV
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N1)

        pid_mask = pid // SPARSE_KV_MULTIPLE

        stride_q_num_blks_h = 16
        stride_q_idx_h = 256
        stride_q_idx_n = 16
        stride_full_q_num_blks_h = 16
        stride_full_q_idx_h = 256
        stride_full_q_idx_n = 16


        dv = tl.zeros([BLOCK_N1, V_HEAD_DIM_ROUNDED], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        start_n1 = pid * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)
        # load K and V: they stay in SRAM throughout the inner loop.
        k = load_checked_2d(K, offs_n1, offs_k, stride_kn, stride_kd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM, INDEX_DTYPE)
        v = load_checked_2d(V, offs_n1, offs_v, stride_vn, stride_vd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM, INDEX_DTYPE)
        if PRESCALE_QK:
            k = (k * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq1 = off_hkv * GQA_SHARED_HEADS + off_g

            # Offset Q, DQ, DO, DELTA & LSE. These inputs are offsetted by query heads.
            q_adj1 = (stride_qh * off_hq1 + stride_qz * off_zq).to(tl.int64)
            do_adj1 = (stride_doh * off_hq1 + stride_doz * off_zq).to(tl.int64)
            dq_adj1 = (stride_dqh * off_hq1 + stride_dqz * off_zq).to(tl.int64)
            off_chz1 = ((off_zq * HQ + off_hq1) * Q_LEN).to(tl.int64)

            Q1 = Q + q_adj1
            DO1 = DO + do_adj1
            # TODO: This does not work if DQ is not the same layout as Q (for example,
            # if Q is broadcasted)
            LSE1 = LSE + off_chz1
            DELTA1 = DELTA + off_chz1

            sparse_idx_hq1 = off_hq1 % SPARSE_HQ
            sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq1

            sparse_q_num_blks_offset = sparse_hz_offset * stride_q_num_blks_h + pid_mask
            sparse_q_idx_offset = sparse_hz_offset * stride_q_idx_h + pid_mask * stride_q_idx_n  # noqa: B950
            sparse_full_q_num_blks_offset = sparse_hz_offset * stride_full_q_num_blks_h + pid_mask
            sparse_full_q_idx_offset = sparse_hz_offset * stride_full_q_idx_h + pid_mask * stride_full_q_idx_n  # noqa: B950

            # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Q_IDX and Q_NUM_BLKS are always contiguous.
            q_indices = Q_IDX + sparse_q_idx_offset
            q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)

            offs_m1 = q_start + tl.arange(0, BLOCK_M1)
            desc_q = None
            desc_do = None
            dk, dv = bwd_dkdv_inner(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
                Q1, DO1, DELTA1, LSE1,
                desc_q, desc_do, q_start,
                dk, dv, k, v,
                off_zq, off_hq1, offs_n1, offs_m1,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=False,
            )


            if HAS_FULL_BLOCKS:
                # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FULL_Q_IDX and FULL_Q_NUM_BLKS are always contiguous.
                q_indices = FULL_Q_IDX + sparse_full_q_idx_offset
                q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_full_q_num_blks_offset)

                offs_m1 = q_start + tl.arange(0, BLOCK_M1)
                dk, dv = bwd_dkdv_inner(
                    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
                    Q1, DO1, DELTA1, LSE1,
                    desc_q, desc_do, q_start,
                    dk, dv, k, v,
                    off_zq, off_hq1, offs_n1, offs_m1,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    q_indices, sparse_q_num_blocks,
                    MATMUL_PRECISION,
                    IS_FULL_BLOCKS=True,
                )

        # Write back dV and dK.
        dv_ptrs = DV + offs_n1[:, None].to(INDEX_DTYPE) * stride_dvm + offs_v[None, :].to(INDEX_DTYPE) * stride_dvd

        index_n = offs_n1[:, None]
        index_k = offs_k[None, :]
        index_v = offs_v[None, :]

        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM))

        dk *= SM_SCALE

        if SAFE_HEAD_DIM:
            mask = index_n < KV_LEN
        else:
            mask = (index_n < KV_LEN) & (index_k < QK_HEAD_DIM)

        # first compute broadcasted dk of shape [Bq, Hkv, KV_LEN, V_HEAD_DIM]
        # then reduce to dk of shape [Bkv, Hkv, KV_LEN, V_HEAD_DIM]
        tl.static_assert(dk.shape == [BLOCK_N1, QK_HEAD_DIM_ROUNDED])
        xindex = index_k + 96*index_n + 196608*off_hkv + 6291456*off_zq
        tl.store(out_ptr0 + (tl.broadcast_to(index_k + 96*off_hkv + 3072*index_n, [BLOCK_N1, QK_HEAD_DIM_ROUNDED])), dk, mask)

@triton.jit
def bwd_dq_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
    K, V, desc_k, desc_v, kv_start, start_m2,
    dq, q, do, Di, lse,
    off_z, off_hq,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):

    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.10206207261596575
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 96
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = False
    USE_TMA : tl.constexpr = False
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = 2048
    KV_LEN = 2048

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)

    hi = tl.minimum(sparse_kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N2), 1))

    kv_offset = 0
    for start_n in range(0, hi):
        dq = bwd_dq_block_mn(
            arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
            dq, q, K, V, desc_k, desc_v, kv_start, kv_offset, start_m2,
            do, Di, lse, Q_LEN, KV_LEN,
            off_z, off_hq, offs_k, offs_v,
            stride_kn, stride_kd, stride_vn, stride_vd,
            kv_indices, sparse_kv_num_blocks,
            MATMUL_PRECISION, RCP_LN2,
            IS_FULL_BLOCKS,
        )

        # Increment pointers.
        offset = get_offset_for_next_block(
            start_n, kv_indices, sparse_kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N2, BLOCKS_ARE_CONTIGUOUS
        )

        kv_offset += offset

    return dq


@triton.jit
def bwd_dq_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
    dq, q, K, V, desc_k, desc_v, kv_start, kv_offset, start_m2,
    do, Di, lse, Q_LEN, KV_LEN,
    off_z, off_hq, offs_k, offs_v,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS,
):
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.10206207261596575
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 96
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = False
    USE_TMA : tl.constexpr = False
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32


    offs_n2 = kv_start + kv_offset + tl.arange(0, BLOCK_N2)
    offs_m2 = start_m2  + tl.arange(0, BLOCK_M2)
    kT_ptrs = K + offs_n2[None, :].to(INDEX_DTYPE) * stride_kn + offs_k[:, None].to(INDEX_DTYPE) * stride_kd
    vT_ptrs = V + offs_n2[None, :].to(INDEX_DTYPE) * stride_vn + offs_v[:, None].to(INDEX_DTYPE) * stride_vd
    # NB reversed order to since K is transposed
    kT = load_checked_2d(kT_ptrs, offs_k, offs_n2, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, QK_HEAD_DIM, KV_LEN, INDEX_DTYPE)
    qk = tl.dot(q, kT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    pre_mod_scores = qk
    n = get_bounded_indices(offs_n2[None, :], KV_LEN if not IS_DIVISIBLE else None)
    # The boundary check is done for the outer loop, but here it's possible since we're iterating across N dim
    # that the M reads out of bounds for the PIDS spanning the Q_LEN boundary
    m = get_bounded_indices(offs_m2[:, None], Q_LEN if not IS_DIVISIBLE else None)

    tmp0 = (qk)
    post_mod_scores = tmp0




    if not IS_DIVISIBLE:
        post_mod_scores = tl.where(offs_n2[None, :] < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp1 = tl.full([1], True, tl.int1)
        tmp2 = (m)
        tmp3 = (n)
        tmp4 = tmp2 >= tmp3
        tmp5 = tmp1 & tmp4
        tmp6 = tl.load(in_ptr16 + tmp2)
        tmp7 = tl.load(in_ptr17 + tmp6)
        tmp8 = tmp3 >= tmp7
        tmp9 = tmp5 & tmp8
        mask_mod_output = tmp9


        # apply mask for partial masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    p = tl.math.exp2(post_mod_scores - lse)
    # Compute dP and dS.
    # NB reversed order to since V is transposed
    vT = load_checked_2d(vT_ptrs, offs_v, offs_n2, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, V_HEAD_DIM, KV_LEN, INDEX_DTYPE)
    dp = tl.dot(do, vT, input_precision=FLOAT32_PRECISION)
    ds = p * (dp - Di[:, None])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    tmp10 = (ds)
    grad_scores = tmp10


    if not IS_DIVISIBLE:
        grad_scores = tl.where(offs_n2[None, :] < KV_LEN, grad_scores, 0.0)

    # ~~~~~~~~~~~~~~~~~~~ Apply other buffer grad writes ~~~~~~~~~~~~~
    if WRITE_DQ:
        scatter_mask = (offs_m2[:, None] < Q_LEN ) & (offs_n2[None, :] < KV_LEN)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = grad_scores

    if not IS_FULL_BLOCKS:
        # (grads) apply mask for partially unmasked block
        ds = tl.where(mask_mod_output, ds, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = ds.to(MATMUL_PRECISION)
    # Compute dQ.
    dq += tl.dot(ds, tl.trans(kT), input_precision=FLOAT32_PRECISION)

    return dq


@triton.jit
def bwd_dkdv_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
    Q, DO, DELTA, LSE, # pointers
    desc_q, desc_do, q_start,
    dk, dv, k, v,
    off_z, off_hq, offs_n1, offs_m1,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.10206207261596575
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 96
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = False
    USE_TMA : tl.constexpr = False
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = 2048
    KV_LEN = 2048

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

    # The minimum is needed to handle the case where we run with a super large
    # SPARSE_BLOCK_SIZE (i.e. no block-mask!)
    hi = tl.minimum(sparse_q_num_blocks * SPARSE_Q_MULTIPLE, tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1))

    offset_block = 0
    for start_m in range(0, hi):
        dk, dv = bwd_dkdv_block_mn(
            arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
            dk, dv, Q, k, v, DO, DELTA, LSE,
            desc_q, desc_do, q_start, offset_block, Q_LEN, KV_LEN,
            off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
            stride_qm, stride_qd, stride_dom, stride_dod,
            q_indices, sparse_q_num_blocks,
            MATMUL_PRECISION, RCP_LN2,
            IS_FULL_BLOCKS,
        )
        # Increment pointers.
        offset = get_offset_for_next_block(
            start_m, q_indices, sparse_q_num_blocks,
            SPARSE_Q_BLOCK_SIZE, SPARSE_Q_MULTIPLE, BLOCK_M1, BLOCKS_ARE_CONTIGUOUS
        )
        offs_m1 += offset
        offset_block += offset

    return dk, dv


@triton.jit
def bwd_dkdv_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, out_ptr0,
    dk, dv, Q, k, v, DO, DELTA, LSE,
    desc_q, desc_do, q_start, offset, Q_LEN, KV_LEN,
    off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS,
):
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.10206207261596575
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 96
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = False
    USE_TMA : tl.constexpr = False
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32

    qT_ptrs = Q + offs_m1[None, :].to(INDEX_DTYPE) * stride_qm + offs_k[:, None].to(INDEX_DTYPE) * stride_qd
    do_ptrs = DO + offs_m1[:, None].to(INDEX_DTYPE) * stride_dom + offs_v[None, :].to(INDEX_DTYPE) * stride_dod
    # NB reversed order since Q is transposed
    qT = load_checked_2d(qT_ptrs, offs_k, offs_m1, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, QK_HEAD_DIM, Q_LEN, INDEX_DTYPE)
    # Load LSE before computing qk to reduce pipeline stall.
    if IS_DIVISIBLE:
        lse = tl.load(LSE + offs_m1)
    else:
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN)
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(k, qT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    m = get_bounded_indices(offs_m1[None, :], Q_LEN if not IS_DIVISIBLE else None)
    # The boundary check is done for the outer loop, but here it's possible since we're iterating across M dim
    # that the n reads out of bounds for the PIDS spanning the KV_LEN boundary
    n = get_bounded_indices(offs_n1[:, None], KV_LEN if not IS_DIVISIBLE else None)

    pre_mod_scores = qkT
    tmp11 = (qkT)
    post_mod_scores = tmp11



    if not IS_DIVISIBLE:
        post_mod_scores = tl.where(offs_m1[None, :] < Q_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp12 = tl.full([1], True, tl.int1)
        tmp13 = (m)
        tmp14 = (n)
        tmp15 = tmp13 >= tmp14
        tmp16 = tmp12 & tmp15
        tmp17 = tl.load(in_ptr16 + tmp13)
        tmp18 = tl.load(in_ptr17 + tmp17)
        tmp19 = tmp14 >= tmp18
        tmp20 = tmp16 & tmp19
        mask_mod_output = tmp20

        # (grads) apply mask for fully masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    pT = tl.math.exp2(post_mod_scores - lse[None, :])
    do = load_checked_2d(do_ptrs, offs_m1, offs_v, None, None, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM, INDEX_DTYPE)
    # Compute dV.
    ppT = pT
    dv += tl.dot(ppT.to(MATMUL_PRECISION), do, input_precision=FLOAT32_PRECISION)
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN)
    # Compute dP and dS.
    dpT = tl.dot(v, tl.trans(do), input_precision=FLOAT32_PRECISION)
    dsT = pT * (dpT - Di[None, :])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    tmp21 = (dsT)
    grad_scores = tmp21



    if not IS_DIVISIBLE:
        grad_scores = tl.where(offs_m1[None, :] < Q_LEN, grad_scores, 0.0)

    # ~~~~~~~~~~~~~~~~~~~ Apply other buffer grad writes ~~~~~~~~~~~~~
    if not WRITE_DQ:
        idx_b = off_z
        idx_h = off_hq
        idx_m = m
        idx_n = n
        scatter_mask = (offs_m1[None, :] < Q_LEN) & (offs_n1[:, None] < KV_LEN)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dsT = grad_scores
    if not IS_FULL_BLOCKS:
        # (grads) apply mask for partially unmasked block
        dsT = tl.where(mask_mod_output, dsT, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dk += tl.dot(dsT.to(MATMUL_PRECISION), tl.trans(qT), input_precision=FLOAT32_PRECISION)

    return dk, dv


# Utility triton funcs
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices

@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_LEN: tl.constexpr,
    INDEX_DTYPE: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None].to(INDEX_DTYPE) * stride_m + offs_n[None, :].to(INDEX_DTYPE) * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_LEN), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_LEN), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, getitem, getitem_1, tangents_1 = args
        args.clear()
        assert_size_stride_grouped((getitem, tangents_1), ((1, 32, 2048, 64), (1, 32, 2048, 64)), ((4194304, 64, 2048, 1), (4194304, 64, 2048, 1)), 'input')
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf1 = empty_strided_cuda((1, 32, 2048), (65536, 2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [flex_attention_backward], Original ATen: [flex_attention_backward]
            raw_stream1 = get_raw_stream(1)
            triton_per_fused_flex_attention_backward_0.run(getitem, tangents_1, buf1, 2048, 32, 64, stream=raw_stream1)
            del getitem
            buf2 = empty_strided_cuda((1, 32, 2048, 96), (6291456, 96, 3072, 1), torch.bfloat16)
            buf3 = empty_strided_cuda((1, 32, 2048, 64), (4194304, 64, 2048, 1), torch.bfloat16)
            assert_size_stride_grouped((primals_1, primals_2, primals_3, getitem_1, primals_5, primals_4, primals_10, primals_11, primals_8, primals_9, primals_12, primals_13, primals_6, primals_7), ((1, 32, 2048, 96), (1, 32, 2048, 96), (1, 32, 2048, 64), (1, 32, 2048), (1, 1, 16), (1, 1, 16, 16), (1, 1, 16), (1, 1, 16, 16), (1, 1, 16), (1, 1, 16, 16), (1, 1, 16), (1, 1, 16, 16), (2048, ), (2176, )), ((3072, 96, 3072, 1), (3072, 96, 3072, 1), (4096, 128, 4096, 1), (65536, 2048, 1), (16, 16, 1), (256, 256, 16, 1), (16, 16, 1), (256, 256, 16, 1), (16, 16, 1), (256, 256, 16, 1), (16, 16, 1), (256, 256, 16, 1), (1, ), (1, )), 'input')
            primals_1 = copy_if_misaligned(primals_1)
            primals_2 = copy_if_misaligned(primals_2)
            primals_3 = copy_if_misaligned(primals_3)
            primals_5 = copy_if_misaligned(primals_5)
            primals_4 = copy_if_misaligned(primals_4)
            primals_10 = copy_if_misaligned(primals_10)
            primals_11 = copy_if_misaligned(primals_11)
            primals_8 = copy_if_misaligned(primals_8)
            primals_9 = copy_if_misaligned(primals_9)
            primals_12 = copy_if_misaligned(primals_12)
            primals_13 = copy_if_misaligned(primals_13)
            primals_6 = copy_if_misaligned(primals_6)
            primals_7 = copy_if_misaligned(primals_7)
            buf4 = empty_strided_cuda((1, 32, 2048, 96), (6291456, 96, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [flex_attention_backward], Original ATen: [flex_attention_backward]
            raw_stream1 = get_raw_stream(1)
            triton_tem_fused_flex_attention_backward_1.run(primals_1, primals_2, primals_3, getitem_1, buf1, tangents_1, buf2, buf3, primals_5, primals_4, primals_10, primals_11, primals_8, primals_9, primals_12, primals_13, primals_6, primals_7, buf4, 64, 1, 32, stream=raw_stream1)
            del buf1
            del getitem_1
            del primals_1
            del primals_10
            del primals_11
            del primals_12
            del primals_13
            del primals_2
            del primals_3
            del primals_4
            del primals_5
            del primals_6
            del primals_7
            del primals_8
            del primals_9
            del tangents_1
        return (buf2, buf4, buf3, None, None, None, None, None, None, None, None, None, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((1, 32, 2048, 96), (3072, 96, 3072, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_2 = rand_strided((1, 32, 2048, 96), (3072, 96, 3072, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 32, 2048, 64), (4096, 128, 4096, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_4 = rand_strided((1, 1, 16, 16), (256, 256, 16, 1), device='cuda:1', dtype=torch.int32)
    primals_5 = rand_strided((1, 1, 16), (16, 16, 1), device='cuda:1', dtype=torch.int32)
    primals_6 = rand_strided((2048, ), (1, ), device='cuda:1', dtype=torch.int32)
    primals_7 = rand_strided((2176, ), (1, ), device='cuda:1', dtype=torch.int32)
    primals_8 = rand_strided((1, 1, 16), (16, 16, 1), device='cuda:1', dtype=torch.int32)
    primals_9 = rand_strided((1, 1, 16, 16), (256, 256, 16, 1), device='cuda:1', dtype=torch.int32)
    primals_10 = rand_strided((1, 1, 16), (16, 16, 1), device='cuda:1', dtype=torch.int32)
    primals_11 = rand_strided((1, 1, 16, 16), (256, 256, 16, 1), device='cuda:1', dtype=torch.int32)
    primals_12 = rand_strided((1, 1, 16), (16, 16, 1), device='cuda:1', dtype=torch.int32)
    primals_13 = rand_strided((1, 1, 16, 16), (256, 256, 16, 1), device='cuda:1', dtype=torch.int32)
    getitem = rand_strided((1, 32, 2048, 64), (4194304, 64, 2048, 1), device='cuda:1', dtype=torch.bfloat16)
    getitem_1 = rand_strided((1, 32, 2048), (65536, 2048, 1), device='cuda:1', dtype=torch.float32)
    tangents_1 = rand_strided((1, 32, 2048, 64), (4194304, 64, 2048, 1), device='cuda:1', dtype=torch.bfloat16)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, getitem, getitem_1, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
