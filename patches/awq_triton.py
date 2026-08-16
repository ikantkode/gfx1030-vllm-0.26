# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

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
    partials = torch.empty((split, N), dtype=torch.float32, device=input.device)
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
        # Gate = the wrapper's asserts, checked here to stay assert-free.
        if K % 16 == 0 and (K // 16) % 32 == 0 and group_size % 32 == 0:
            return awq_gemv_splitk_triton(
                input, qweight, scales, qzeros, 64, 32, 16, 2, 3,
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
        block_size_m, block_size_n, block_size_k = 16, 128, 64
        num_warps, num_stages = 8, 3
        split_k_iters = 1 if K <= 4096 else 8

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
