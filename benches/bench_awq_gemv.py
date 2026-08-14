"""AWQ GEMV kernel prototype for M=1 decode on gfx1030 (RDNA2).

No tl.dot (no wasted M-tile): each program owns BLOCK_N output columns,
streams BLOCK_K k-chunks of packed int4 weights, dequantizes (identical
interleave/shifts logic to the stock kernel), and reduces with tl.sum.
Constraint: group_size (128) % BLOCK_K == 0 so each k-chunk sits in one group.

Bench: correctness vs the stock dot-kernel (current best configs), then sweep.
"""

import time

import torch
import triton
import triton.language as tl


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
    pid = tl.program_id(0)

    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    masks_n = offs_n < N
    offs_n8 = pid * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    masks_n8 = offs_n8 < N // 8

    # reverse AWQ order [0,4,1,5,2,6,3,7] -> nibble shifts
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

        # whole BLOCK_K chunk lies within one quant group (enforced by caller)
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
        w = (b - zeros) * scales  # fp16

        acc += tl.sum(
            w.to(tl.float32) * x.to(tl.float32)[:, None], axis=0
        )

    out = acc.to(out_ptr.type.element_ty)
    tl.store(out_ptr + offs_n, out, mask=masks_n)


def gemv(x, q, s, z, m, k, BN, BK, W, S, g=128):
    out = torch.empty(1, m, dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(m, BN),)
    awq_gemv_kernel[grid](
        x, q, out, z, s, k, m, g,
        BLOCK_N=BN, BLOCK_K=BK, num_warps=W, num_stages=S,
    )
    return out


# ---------------- reference: current best dot-kernel timings ----------------
DOT_BEST = {(9216, 2560): 92.8, (2560, 9216): 103.1, (2560, 2560): None}

print("=== correctness + sweep: GEMV vs dot-kernel ===", flush=True)
for (m, k) in [(9216, 2560), (2560, 9216), (2560, 2560)]:
    torch.manual_seed(42)
    q = torch.randint(-2**31, 2**31 - 1, (k, m // 8), dtype=torch.int32, device="cuda")
    s = torch.randn(k // 128, m, dtype=torch.float16, device="cuda") * 0.01
    z = torch.randint(-2**31, 2**31 - 1, (k // 128, m // 8), dtype=torch.int32, device="cuda")
    x = torch.randn(1, k, dtype=torch.float16, device="cuda")

    # reference via vllm's patched wrapper (fp16 accumulate)
    from vllm.model_executor.layers.quantization.awq_triton import awq_gemm_triton
    ref = awq_gemm_triton(x, q, s, z, 8).float()

    wbytes = k * m * 0.5 + k * m / 128 * 2 + k * m / (8 * 128) * 4
    floor = wbytes / 400e9 * 1e6
    print(f"--- m={m} k={k} (floor {floor:.1f}us, dot-best {DOT_BEST[(m,k)]}us) ---", flush=True)

    for (BN, BK, W, S) in [
        (128, 64, 4, 3), (128, 128, 4, 3), (64, 64, 4, 3), (64, 128, 2, 3),
        (256, 64, 8, 3), (64, 64, 2, 3), (128, 64, 8, 3), (32, 128, 4, 3),
    ]:
        if 128 % BK != 0:
            continue
        try:
            o = gemv(x, q, s, z, m, k, BN, BK, W, S)
            torch.cuda.synchronize()
            err = (o.float() - ref).abs().max().item()
            rel = err / (ref.abs().max().item() + 1e-9)
            for _ in range(10):
                gemv(x, q, s, z, m, k, BN, BK, W, S)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(200):
                gemv(x, q, s, z, m, k, BN, BK, W, S)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t) / 200 * 1e6
            flag = "OK " if rel < 0.02 else "BAD"
            print(f"GEMV BN={BN:3d} BK={BK:3d} W={W} S={S}: {us:7.1f} us "
                  f"({floor / us * 100:4.0f}% peak)  max_err={err:.3f} rel={rel:.4f} [{flag}]", flush=True)
        except Exception as e:
            print(f"GEMV BN={BN} BK={BK} W={W} S={S}: FAIL {type(e).__name__} {str(e)[:80]}", flush=True)
    del q, s, z, x, ref
    torch.cuda.empty_cache()
