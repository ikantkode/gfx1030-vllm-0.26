"""Split-K GEMV variant: grid (N-blocks, SK), atomic_add accumulation."""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def awq_gemv_sk_kernel(
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
    SK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

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

    # this program's K range: [pid_k * ceil(K/SK), ...) in BLOCK_K steps,
    # striding SK blocks so group alignment is preserved per k0
    k_per_split = tl.cdiv(K, SK)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        masks_k = offs_k < k_end

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
        w = (b - zeros) * scales

        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[:, None], axis=0)

    tl.atomic_add(out_ptr + offs_n, acc.to(out_ptr.type.element_ty), mask=masks_n)


def run(x, q, s, z, m, k, BN, BK, SK, W=4, S=3, g=128):
    out = torch.zeros(1, m, dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(m, BN), SK)
    awq_gemv_sk_kernel[grid](
        x, q, out, z, s, k, m, g,
        BLOCK_N=BN, BLOCK_K=BK, SK=SK, num_warps=W, num_stages=S,
    )
    return out


print("=== split-K GEMV sweep (atomic accumulation) ===", flush=True)
for (m, k, base) in [(2560, 9216, 94.2), (9216, 2560, 58.4)]:
    torch.manual_seed(7)
    q = torch.randint(-2**31, 2**31 - 1, (k, m // 8), dtype=torch.int32, device="cuda")
    s = torch.randn(k // 128, m, dtype=torch.float16, device="cuda") * 0.01
    z = torch.randint(-2**31, 2**31 - 1, (k // 128, m // 8), dtype=torch.int32, device="cuda")
    x = torch.randn(1, k, dtype=torch.float16, device="cuda")
    from vllm.model_executor.layers.quantization.awq_triton import awq_gemv_triton
    ref = awq_gemv_triton(x, q, s, z, 128 if m >= 4096 else 32, 64 if m >= 4096 else 128).float()
    wbytes = k * m * 0.5 + k * m / 128 * 2 + k * m / 1024
    floor = wbytes / 400e9 * 1e6
    print(f"--- m={m} k={k} (SK1-best {base}us, floor {floor:.1f}us) ---", flush=True)
    for (BN, BK, SK) in [
        (32, 128, 1), (32, 128, 2), (32, 128, 4), (32, 128, 8),
        (64, 128, 4), (64, 64, 4), (16, 128, 4),
    ]:
        if m == 9216 and BN < 64:
            continue
        try:
            o = run(x, q, s, z, m, k, BN, BK, SK)
            torch.cuda.synchronize()
            rel = ((o.float() - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
            for _ in range(10):
                run(x, q, s, z, m, k, BN, BK, SK)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(200):
                run(x, q, s, z, m, k, BN, BK, SK)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t) / 200 * 1e6
            print(f"SK-GEMV BN={BN:3d} BK={BK:3d} SK={SK}: {us:7.1f} us ({floor / us * 100:4.0f}% peak)  rel={rel:.4f}", flush=True)
        except Exception as e:
            print(f"SK-GEMV BN={BN} BK={BK} SK={SK}: FAIL {type(e).__name__} {str(e)[:80]}", flush=True)
    del q, s, z, x, ref
    torch.cuda.empty_cache()
