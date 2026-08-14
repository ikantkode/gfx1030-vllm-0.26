"""Standalone sweep of the vLLM AWQ triton gemm kernel for gfx1030 decode (M=1).

Copies awq_gemm_kernel verbatim (SPLIT_K=1 path) and sweeps
(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) against the bandwidth floor.
Run inside the container: docker exec -i qwen-vllm python3 -u - < bench_awq.py
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def awq_gemm_kernel(
    a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K, group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    accumulator_dtype = c_ptr.type.element_ty
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

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

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)
        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

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


def bench(m, k, BM, BN, BK, W, S, g=128):
    N = m
    q = torch.randint(-2**31, 2**31 - 1, (k, m // 8), dtype=torch.int32, device="cuda")
    s = torch.randn(k // g, m, dtype=torch.float16, device="cuda") * 0.01
    z = torch.randint(-2**31, 2**31 - 1, (k // g, m // 8), dtype=torch.int32, device="cuda")
    x = torch.randn(1, k, dtype=torch.float16, device="cuda")
    out = torch.empty((1, m), dtype=torch.float16, device="cuda")
    M = 1
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), 1)
    def call():
        awq_gemm_kernel[grid](
            x, q, out, z, s, M, N, k, g,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, SPLIT_K=1,
            num_warps=W, num_stages=S,
        )
    try:
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            call()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t) / 200 * 1e6
        wbytes = k * m * 0.5 + k * m / g * 2 + k * m / (8 * g) * 4
        floor = wbytes / 400e9 * 1e6
        print(f"m={m:5d} k={k:5d} BM={BM:3d} BN={BN:3d} BK={BK:3d} W={W} S={S}: "
              f"{us:7.1f} us  ({floor / us * 100:4.0f}% of peak bw)", flush=True)
        return us
    except Exception as e:
        print(f"m={m} k={k} BM={BM} BN={BN} BK={BK} W={W} S={S}: FAIL {type(e).__name__} {str(e)[:80]}", flush=True)
        return None


CONFIGS = [
    (32, 32, 32, 4, 3),   # current (post SPLIT_K=1 patch)
    (16, 64, 64, 4, 3),
    (16, 128, 64, 8, 3),
    (32, 64, 64, 4, 3),
    (32, 64, 64, 8, 3),
    (32, 128, 64, 8, 3),
    (32, 128, 128, 8, 4),
    (32, 256, 64, 8, 3),
    (16, 128, 128, 8, 4),
    (32, 128, 32, 8, 3),
]

for shape in [(9216, 2560), (2560, 9216)]:
    print(f"--- shape m,k={shape} ---", flush=True)
    for cfg in CONFIGS:
        bench(*shape, *cfg)
