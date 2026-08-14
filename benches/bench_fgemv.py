"""fp16 GEMV (M=1) Triton kernel for gfx1030 — targets the k>8192 LLMM1 hole."""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def fp16_gemv_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    M,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    masks_m = offs_m < M
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        masks_k = offs_k < K
        x = tl.load(x_ptr + offs_k, mask=masks_k, other=0.0)
        w = tl.load(
            w_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=masks_m[:, None] & masks_k[None, :],
            other=0.0,
        )
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(o_ptr + offs_m, acc.to(o_ptr.type.element_ty), mask=masks_m)


def gemv16(x, w, BN, BK, W=4, S=3):
    M = w.shape[0]
    K = w.shape[1]
    out = torch.empty(1, M, dtype=torch.float16, device="cuda")
    grid = (triton.cdiv(M, BN),)
    fp16_gemv_kernel[grid](x, w, out, M, K, BLOCK_N=BN, BLOCK_K=BK, num_warps=W, num_stages=S)
    return out


print("=== fp16 GEMV sweep (target: m=2560 k=9216, rocBLAS=704us in-table) ===", flush=True)
for (m, k) in [(2560, 9216), (8192, 2560)]:
    torch.manual_seed(3)
    w = torch.randn(m, k, dtype=torch.float16, device="cuda")
    x = torch.randn(1, k, dtype=torch.float16, device="cuda")
    ref = torch.nn.functional.linear(x, w)
    floor = m * k * 2 / 400e9 * 1e6
    print(f"--- m={m} k={k} (floor {floor:.0f}us) ---", flush=True)
    t = time.perf_counter()
    for _ in range(50):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    print(f"  rocBLAS: {(time.perf_counter()-t)/50*1e6:7.1f} us", flush=True)
    for (BN, BK, W) in [(32, 64, 4), (32, 128, 4), (64, 128, 4), (16, 128, 4), (64, 256, 8), (32, 256, 8)]:
        try:
            o = gemv16(x, w, BN, BK, W)
            torch.cuda.synchronize()
            rel = ((o.float() - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
            for _ in range(10):
                gemv16(x, w, BN, BK, W)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(100):
                gemv16(x, w, BN, BK, W)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t) / 100 * 1e6
            print(f"  GEMV16 BN={BN:3d} BK={BK:3d} W={W}: {us:7.1f} us ({floor/us*100:3.0f}% peak) rel={rel:.4f}", flush=True)
        except Exception as e:
            print(f"  GEMV16 BN={BN} BK={BK} W={W}: FAIL {type(e).__name__} {str(e)[:70]}", flush=True)
    del w, x, ref
    torch.cuda.empty_cache()
