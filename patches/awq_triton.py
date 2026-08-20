# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

# Rung 13: fused Gemma RMSNorm for gfx1030 — imported here so every process
# that loads the AWQ kernels (incl. the V1 engine child) applies the patch.
import vllm.model_executor.layers.rmsnorm_gfx1030  # noqa: F401

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,  # quantized matrix
    scales_ptr,  # scales, per group
    zeros_ptr,  # zeros, per group
    group_size,  # Should always be one of the supported group sizes
    result_ptr,  # Output matrix
    num_cols,  # input num cols in qweight
    num_rows,  # input num rows in qweight
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Set up the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = 8 * num_cols * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


@triton.jit
def awq_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    M,
    N,
    K,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = c_ptr.type.element_ty

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :],
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < N // 8

    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < N // 8

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
    # block_offset = BLOCK_SIZE_K * SPLIT_K
    # for k in range(0, (K + block_offset - 1) // (block_offset)):
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Dequantize b.
        offsets_szk = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
        ) // group_size + tl.arange(0, 1)
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def awq_gemv_kernel(
    x_ptr,
    qweight_ptr,
    out_ptr,
    zeros_ptr,
    scales_ptr,
    K,
    N,
    group_size,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # [gfx1030/RDNA2] Single-token (M==1) GEMV: no tl.dot, so no wasted M-tile.
    # Each program owns BLOCK_N output columns and streams the packed int4
    # weights for BLOCK_K rows per iteration (must lie within one quant group:
    # caller guarantees group_size % BLOCK_K == 0).
    pid = tl.program_id(0)

    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    masks_n = offs_n < N
    offs_n8 = pid * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    masks_n8 = offs_n8 < N // 8

    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_K * (BLOCK_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_K, BLOCK_N))

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        masks_k = offs_k < K

        x = tl.load(x_ptr + offs_k, mask=masks_k, other=0.0)

        offs_b = (N // 8) * offs_k[:, None] + offs_n8[None, :]
        b = tl.load(
            qweight_ptr + offs_b,
            mask=masks_k[:, None] & masks_n8[None, :],
            other=0.0,
        )
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        g_row = k0 * BLOCK_K // group_size + tl.arange(0, 1)
        offs_z = (N // 8) * g_row[:, None] + offs_n8[None, :]
        zeros = tl.load(
            zeros_ptr + offs_z,
            mask=(g_row[:, None] < K // group_size) & masks_n8[None, :],
            other=0.0,
        )
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_K, BLOCK_N))

        offs_s = N * g_row[:, None] + offs_n[None, :]
        scales = tl.load(
            scales_ptr + offs_s,
            mask=(g_row[:, None] < K // group_size) & masks_n[None, :],
            other=0.0,
        )
        scales = tl.broadcast_to(scales, (BLOCK_K, BLOCK_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        w = (b - zeros) * scales

        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[:, None], axis=0)

    out = acc.to(out_ptr.type.element_ty)
    tl.store(out_ptr + offs_n, out, mask=masks_n)


# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(
        qweight.shape[0],
        qweight.shape[1] * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )
    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result


# [gfx1030] single-token GEMV wrapper: input [1, K], qweight [K, N//8],
# scales [K//G, N], qzeros [K//G, N//8]. BLOCK_K must divide group_size.
def awq_gemv_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]
    assert group_size % block_k == 0
    result = torch.empty((M, N), dtype=scales.dtype, device=input.device)
    grid = (triton.cdiv(N, block_n),)
    awq_gemv_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        K,
        N,
        group_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return result


# [gfx1030] K-split GEMV for latency-bound small-N shapes. Rung-9 sweep fact:
# N<=4096 shapes run at 8-27% of peak BW because cdiv(N, BN) programs cannot
# fill ~80 CUs. Splitting K along axis 1 of the grid multiplies program count;
# each program reduces its K-slice into an fp32 partial row, and a tiny second
# kernel sums the SPLIT partials into the fp16 output.
@triton.jit
def awq_gemv_splitk_kernel(
    x_ptr,
    qweight_ptr,
    partials_ptr,
    zeros_ptr,
    scales_ptr,
    K,
    N,
    group_size,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    masks_n = offs_n < N
    offs_n8 = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    masks_n8 = offs_n8 < N // 8

    # reverse AWQ order [0,4,1,5,2,6,3,7] -> nibble shifts
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_K * (BLOCK_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_K, BLOCK_N))

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # caller guarantees K % SPLIT == 0 and (K // SPLIT) % BLOCK_K == 0, so
    # every chunk stays K-tail-free and within one quant group (BLOCK_K | 128).
    k_per = K // SPLIT
    k_lo = pid_k * k_per
    for k0 in range(k_lo, k_lo + k_per, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k)

        offs_b = (N // 8) * offs_k[:, None] + offs_n8[None, :]
        b = tl.load(qweight_ptr + offs_b, mask=masks_n8[None, :], other=0.0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        g_row = k0 // group_size + tl.arange(0, 1)
        offs_z = (N // 8) * g_row[:, None] + offs_n8[None, :]
        zeros = tl.load(
            zeros_ptr + offs_z,
            mask=(g_row[:, None] < K // group_size) & masks_n8[None, :],
            other=0.0,
        )
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_K, BLOCK_N))

        offs_s = N * g_row[:, None] + offs_n[None, :]
        scales = tl.load(
            scales_ptr + offs_s,
            mask=(g_row[:, None] < K // group_size) & masks_n[None, :],
            other=0.0,
        )
        scales = tl.broadcast_to(scales, (BLOCK_K, BLOCK_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        w = (b - zeros) * scales  # fp16

        acc += tl.sum(
            w.to(tl.float32) * x.to(tl.float32)[:, None], axis=0
        )

    tl.store(partials_ptr + pid_k * N + offs_n, acc, mask=masks_n)


@triton.jit
def awq_gemv_splitk_reduce(
    partials_ptr,
    out_ptr,
    N,
    SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    masks = offs < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(SPLIT):
        acc += tl.load(partials_ptr + s * N + offs, mask=masks, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.type.element_ty), mask=masks)


# [gfx1030] K-split GEMV wrapper. Requires K % split == 0 and
# (K // split) % block_k == 0 (block_k must still divide group_size).
#
# [gfx1030 rung 11] Persistent partials scratch. Per-call torch.empty for the
# (split, N) fp32 buffer costs ~200 allocs/token in eager segments; a cache
# keyed by (N, split, device) allocates once and reuses. Graph-safe: the
# buffer is allocated on first forward (eager warmup or first capture pass)
# and its address is then as stable as a weight tensor's — captured kernels
# keep pointing at it. Same-stream sequential decode guarantees the previous
# call's reduce consumed the partials before the next gemv overwrites them.
# NOTE: only the INTERNAL partials are cached. `result` must stay per-call —
# gate/up share an (N, K) key, so a cached output would clobber gate's result
# before silu_and_mul reads both. Total cache size here: ~1.8 MB (6 shapes).
_SPLITK_PARTIALS: dict = {}


def _splitk_partials(N: int, split: int, device: torch.device) -> torch.Tensor:
    key = (N, split, device.index)
    buf = _SPLITK_PARTIALS.get(key)
    if buf is None:
        buf = torch.empty((split, N), dtype=torch.float32, device=device)
        _SPLITK_PARTIALS[key] = buf
    return buf


def awq_gemv_splitk_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    block_n: int = 16,
    block_k: int = 128,
    split: int = 8,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]
    assert group_size % block_k == 0
    assert K % split == 0 and (K // split) % block_k == 0
    partials = _splitk_partials(N, split, input.device)
    grid = (triton.cdiv(N, block_n), split)
    awq_gemv_splitk_kernel[grid](
        input,
        qweight,
        partials,
        qzeros,
        scales,
        K,
        N,
        group_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT=split,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    result = torch.empty((M, N), dtype=scales.dtype, device=input.device)
    awq_gemv_splitk_reduce[(triton.cdiv(N, 256),)](
        partials,
        result,
        N,
        SPLIT=split,
        BLOCK_N=256,
        num_warps=4,
    )
    return result


# [gfx1030] Rung 17: FUSED K-split GEMV. The rung-15 splitk path pays a whole
# second kernel launch + a (SPLIT, N) fp32 round-trip + a cdiv(N,256) reduce
# launch for EVERY one of the 120 decode GEMV calls/token. On a 12.64 ms
# token that's ~360 launches of a ~2us kernel + a 1.7 GB fp32 scratch traffic
# the roofline never counted. Fuse the reduce into the splitk kernel: zero the
# fp16 out, then each (pid_n, pid_k) program atomic-adds its fp16 K-slice
# partial straight into out. One kernel, one launch, no partials buffer, no
# reduce. fp16 atomic add is exact (no overflow: 12-bit mantissa, |partial|
# bounded by the fp16 weight/activation range) and the intra-program acc stays
# fp32 so precision is unchanged vs the split-then-sum path (both accumulate
# fp32 per K-slice, differ only in the final fp16 add order).
@triton.jit
def awq_gemv_splitk_fused_kernel(
    x_ptr,
    qweight_ptr,
    out_ptr,
    zeros_ptr,
    scales_ptr,
    K,
    N,
    group_size,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    masks_n = offs_n < N
    offs_n8 = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    masks_n8 = offs_n8 < N // 8

    # out is pre-zeroed by the wrapper (torch.zeros); the SPLIT programs for a
    # given pid_n atomic-add into the same fp16 row-band and the atomic
    # serializes them, so no cross-program zero ordering is needed in-kernel.

    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_K * (BLOCK_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_K, BLOCK_N))

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    k_per = K // SPLIT
    k_lo = pid_k * k_per
    for k0 in range(k_lo, k_lo + k_per, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k)

        offs_b = (N // 8) * offs_k[:, None] + offs_n8[None, :]
        b = tl.load(qweight_ptr + offs_b, mask=masks_n8[None, :], other=0.0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        g_row = k0 // group_size + tl.arange(0, 1)
        offs_z = (N // 8) * g_row[:, None] + offs_n8[None, :]
        zeros = tl.load(
            zeros_ptr + offs_z,
            mask=(g_row[:, None] < K // group_size) & masks_n8[None, :],
            other=0.0,
        )
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_K, BLOCK_N))

        offs_s = N * g_row[:, None] + offs_n[None, :]
        scales = tl.load(
            scales_ptr + offs_s,
            mask=(g_row[:, None] < K // group_size) & masks_n[None, :],
            other=0.0,
        )
        scales = tl.broadcast_to(scales, (BLOCK_K, BLOCK_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        w = (b - zeros) * scales  # fp16

        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[:, None], axis=0)

    # fused reduce: fp32 acc -> fp16, atomic-add straight into out. No separate
    # partials buffer, no reduce kernel. out was pre-zeroed by the wrapper.
    tl.atomic_add(out_ptr + offs_n, acc.to(out_ptr.type.element_ty), mask=masks_n)


def awq_gemv_splitk_fused_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    block_n: int = 16,
    block_k: int = 128,
    split: int = 8,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]
    assert group_size % block_k == 0
    assert K % split == 0 and (K // split) % block_k == 0
    # out is the accumulation target: must start at 0 so the atomic adds
    # sum to the true result. torch.zeros is one extra launch (memset) but
    # far cheaper than the rung-15 (partials alloc + reduce kernel) pair.
    out = torch.zeros((M, N), dtype=scales.dtype, device=input.device)
    grid = (triton.cdiv(N, block_n), split)
    awq_gemv_splitk_fused_kernel[grid](
        input,
        qweight,
        out,
        qzeros,
        scales,
        K,
        N,
        group_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT=split,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    # [gfx1030/RDNA2 decode tuning] For skinny matmuls (small M == decode), use
    # the swept-best tiles BM=16/BN=128/BK=64/W=8/S=3 and pick SPLIT_K by K:
    # short K (<=4096): SK=1 (no partial/reduction overhead; SK=2 tied in sweep)
    # long  K (> 4096): SK=8 (K-split occupancy; 103us vs 220us on m=2560,k=9216)
    num_warps, num_stages = 4, 3
    if M == 1:
        # [gfx1030] Rung 10: K-split GEMV first. Rung 9 proved the small-N
        # shapes latency-bound (8-27% of peak BW; cdiv(N,BN) programs can't
        # fill ~80 CUs), so add a K-split grid axis + fp32 partials + tiny
        # reduce. Full-sweep winner is ONE config for all six -vd shapes:
        #   BN=64 BK=32 SP=16 W=2 S=3  (bench_awq5.py, GPU 1, L2-flushed
        #   median-60, correctness-gated vs the rung-9 dispatch):
        #   N=9216 K=2560 (x64): 66.3 -> 55.3us
        #   N=8192 K=2560 (x32): 69.2 -> 54.2us
        #   N=4096 K=2560 (x24): 55.2 -> 35.0us
        #   N=2560 K=9216 (x32): 102.8 -> 51.8us
        #   N=2560 K=4096 (x32): 52.4 -> 32.3us
        #   N=1024 K=2560 (x16): 36.3 -> 18.7us
        # INT4 GEMV block 13.33 -> 9.10 ms/token; projected ~69 TPS.
        # [gfx1030] Rung 15: the rung-9/10 shape table above is STALE - the
        # true -vd decode GEMV set is 120 calls/token over 5 shapes (1.74 GB;
        # probe_sk2.py, GPU 1, live-config capture): (N,K) x count
        #   (18432,2560) x30 (fused gate+up)  (12288,2560) x22
        #   (10240,2560) x8  (2560,9216) x30 (MLP down)  (2560,4096) x30
        # Re-swept those true shapes (probe_sk2 sweep, L2-flushed cold med-30,
        # rel<0.02 vs live-cfg ref): the three big-N K=2560 shapes all want
        # BN=128 (64 B/row contiguous reads vs 32 B at BN=64) + SP=4/W=4:
        #   (18432,2560): 92.9 -> 78.8us (-0.422ms/token)
        #   (12288,2560): 69.7 -> 60.4us (-0.204)
        #   (10240,2560): 60.1 -> 53.0us (-0.057)
        #   (2560,9216) : 52.2 -> 50.9us at 128/64/16/W2 (-0.040, marginal)
        #   (2560,4096) : 32.4 ~ 32.3us, keep live cfg (latency-bound, 37% peak)
        # Splitk block 7.34 -> 6.62 ms; decode 12.64 -> ~11.9 ms = ~84 TPS.
        # Gate = the wrapper's asserts, checked here to stay assert-free.
        if K % 16 == 0 and (K // 16) % 32 == 0 and group_size % 32 == 0:
            bn, bk, sp, w = {
                # [gfx1030] Rung 15 (SHIPPED v1.1.0). Phase 4 (Rung 19) tried an
                # SP=1 (no K-split) re-tune of the two big gate/up shapes based on
                # a two-run synthetic config sweep (bench_p4_gemv.py), but the
                # in-situ gate on the REAL -vd model (gate_r19.py) showed it was
                # NOT a win: (18432,2560) (256,32,1,2) ran 200.6us vs 181.3us at
                # the shipped (128,32,4,4) -- SLOWER. The synthetic sweep's
                # absolute scale is ~4.7x off the in-situ value and, in-situ,
                # every decode GEMV shape is latency-floor-bound (~180-200us
                # regardless of config), so config choice is a wash and the
                # micro-bench deltas were noise. Reverted to rung-15 configs.
                # See PROGRESS.md Entry 47.
                (18432, 2560): (128, 32, 4, 4),
                (12288, 2560): (128, 32, 4, 4),
                (10240, 2560): (128, 32, 4, 4),
                (2560, 9216): (128, 64, 16, 2),
            }.get((N, K), (64, 32, 16, 2))
            return awq_gemv_splitk_triton(
                input, qweight, scales, qzeros, bn, bk, sp, w, 3,
            )
        # [gfx1030] dedicated GEMV path (no tl.dot M-tile waste). Rung 9:
        # re-swept over the full -vd shape set (6 unique shapes, 135-config
        # grid, L2-flushed cold timing, correctness-gated vs the dot kernel).
        # Per-shape winners (old heuristic -> new, us):
        #   N=9216 K=2560 (x64): 128/64/4/3 67.0 -> 128/128/4/3 66.2
        #   N=8192 K=2560 (x32): 128/64/4/3 69.1 -> unchanged
        #   N=4096 K=2560 (x24): 128/64/4/3 60.4 -> 128/128/4/3 55.4
        #   N=2560 K=9216 (x32): 32/128/4/3 118.2 -> 16/128/8/3 103.0
        #   N=2560 K=4096 (x32): 32/128/4/3 59.5 -> 16/128/8/3 52.4
        #   N=1024 K=2560 (x16): 32/128/4/3 43.6 -> 16/128/8/3 36.1
        # Rule: large N keeps BN=128/W4; small N wants BN=16 (more programs)
        # + W8 (latency hiding). Superseded per-shape by splitk above; kept
        # as the fallback for shapes failing the splitk divisibility gate.
        gemv_cfg = {
            (9216, 2560): (128, 128, 4, 3),
            (8192, 2560): (128, 64, 4, 3),
            (4096, 2560): (128, 128, 4, 3),
            (2560, 9216): (16, 128, 8, 3),
            (2560, 4096): (16, 128, 8, 3),
            (1024, 2560): (16, 128, 8, 3),
        }.get((N, K))
        if gemv_cfg is None:
            # unseen shape: old heuristic fallback (never worse than stock)
            block_n = 128 if N >= 4096 else 32
            block_k = 64 if block_n == 128 else 128
            gemv_cfg = (block_n, block_k, 4, 3)
        return awq_gemv_triton(
            input, qweight, scales, qzeros,
            gemv_cfg[0], gemv_cfg[1], gemv_cfg[2], gemv_cfg[3],
        )
    if M <= 32:
        # [gfx1030] Rung 18: M<=32 tiled GEMM (the 3-32-user decode band).
        # Stock was a single (BM16,BN128,BK64,W8,S3,SK 1/8-by-K) config for ALL
        # M<=32 - never swept. Focused sweep at M in {8,16,32} (the mns=8/16
        # decode-M regimes), real captured weights, in-process, L2-flushed
        # cold med-15, fp16 matmul_close(atol=.8,rtol=.05) gate vs the triton
        # dequant x fp32 matmul: all 15 (shape x M) winners PASS.
        #   Per-(N,K) winners keyed by M-band. M is the #concurrent decodes;
        #   for M in (3,32] not exactly tested, use the smallest band at-or-
        #   ABOVE M (a BM tile covers up to BM rows: BM=8 covers M<=8, BM=16
        #   covers 9-16, BM=32 covers 17-32).
        #   (N,K)         M-band -> (BM,BN,BK,W,S,SK)
        #   (18432,2560)  8:(8,128,32,8,3,1) 16:(16,128,64,8,2,1) 32:(32,128,64,8,3,1)
        #   (12288,2560)  8:(8,128,32,8,3,1) 16:(16,128,32,8,2,1) 32:(32,128,32,8,2,1)
        #   (10240,2560)  8:(8,128,32,8,2,1) 16:(16,128,32,8,2,4) 32:(32,128,32,8,2,1)
        #   (2560,9216)   8:(8,128,32,8,2,4) 16:(16,128,32,8,2,8) 32:(32,128,32,8,2,8)
        #   (2560,4096)   8:(8,128,32,8,3,4) 16:(16,128,32,8,3,4) 32:(32,128,32,8,2,8)
        _R18 = {
            (18432, 2560): {(8,): (8, 128, 32, 8, 3, 1),
                            (16,): (16, 128, 64, 8, 2, 1),
                            (32,): (32, 128, 64, 8, 3, 1)},
            (12288, 2560): {(8,): (8, 128, 32, 8, 3, 1),
                            (16,): (16, 128, 32, 8, 2, 1),
                            (32,): (32, 128, 32, 8, 2, 1)},
            (10240, 2560): {(8,): (8, 128, 32, 8, 2, 1),
                            (16,): (16, 128, 32, 8, 2, 4),
                            (32,): (32, 128, 32, 8, 2, 1)},
            (2560, 9216):  {(8,): (8, 128, 32, 8, 2, 4),
                            (16,): (16, 128, 32, 8, 2, 8),
                            (32,): (32, 128, 32, 8, 2, 8)},
            (2560, 4096):  {(8,): (8, 128, 32, 8, 3, 4),
                            (16,): (16, 128, 32, 8, 3, 4),
                            (32,): (32, 128, 32, 8, 2, 8)},
        }
        _shape_tbl = _R18.get((N, K))
        if _shape_tbl is not None:
            # pick the smallest tested band AT-OR-ABOVE M (a BM tile covers up
            # to BM rows, so M=9-16 -> BM16, M=17-32 -> BM32, M<=8 -> BM8).
            _ge = [b for (b,) in _shape_tbl if b >= M]
            _band = min(_ge) if _ge else max(b for (b,) in _shape_tbl)
            (block_size_m, block_size_n, block_size_k,
             num_warps, num_stages, split_k_iters) = _shape_tbl[(_band,)]
        else:
            # unseen M<=32 shape: old stock config (never worse than before)
            block_size_m, block_size_n, block_size_k = 16, 128, 64
            num_warps, num_stages = 8, 3
            split_k_iters = 1 if K <= 4096 else 8
    else:
        # [gfx1030] Rung 16: M>32 tiled GEMM was LEFT AT SIGNATURE DEFAULTS
        # (BM32/BN32/BK32 W4 S3 + caller's split_k) - the multi-user ladder's
        # "knee" (16+ seqs saturating) lives in this branch. Focused sweep at
        # M=128 (the seqs=128 decode-M regime), 144-config bounded grid
        # (excl. pathological bm128/bn256), L2-flushed cold med-15, fp16
        # matmul_close(atol=.8,rtol=.05) gate vs the triton dequant x fp32
        # matmul: ONE tile wins ALL FIVE -vd shapes, only split-K varies.
        #   (N,K)      winner (BM,BN,BK,W,S,SK)   best_us  stock_us  gain
        #   (18432,2560)x30  64,128,32,8,2,sk1    766.4   1666.6   2.17x
        #   (12288,2560)x22  64,128,32,8,2,sk4    565.3   1109.2   1.96x
        #   (10240,2560)x8   64,128,32,8,2,sk4    462.8    908.6   1.96x
        #   (2560,9216) x30  64,128,32,8,2,sk8    417.9    803.3   1.92x
        #   (2560,4096) x30  64,128,32,8,2,sk8    206.9    366.4   1.77x
        #   AGG 57878.7 vs 116760.2 us = 2.02x on the per-decode-step AWQ
        #   block (57 -> 115 GB/s, vs 438 GB/s pure-read peak = Entry 36).
        # NOTE: sk is NOT a clean K-threshold - the two MIDDLE K=2560 shapes
        # want sk4 while the largest K=2560 shape wants sk1, so key the exact
        # per-shape (N,K) like the M==1 / M<=32 tables, not a K cutoff.
        (block_size_m, block_size_n, block_size_k,
         num_warps, num_stages, split_k_iters) = {
            (18432, 2560): (64, 128, 32, 8, 2, 1),
            (12288, 2560): (64, 128, 32, 8, 2, 4),
            (10240, 2560): (64, 128, 32, 8, 2, 4),
            (2560, 9216):  (64, 128, 32, 8, 2, 8),
            (2560, 4096):  (64, 128, 32, 8, 2, 8),
        }.get((N, K), (64, 128, 32, 8, 2, 8))  # unseen M>32: best default

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    if split_k_iters == 1:
        # No K-split: write directly into a single (M, N) output; no reduction.
        result = torch.empty((M, N), dtype=scales.dtype, device=input.device)
        awq_gemm_kernel[grid](
            input,
            qweight,
            result,
            qzeros,
            scales,
            M,
            N,
            K,
            group_size,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
            SPLIT_K=split_k_iters,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return result

    result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    awq_gemm_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        M,
        N,
        K,
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k_iters,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    result = result.sum(0)

    return result
