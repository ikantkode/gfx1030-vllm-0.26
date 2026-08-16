"""Offline gate for rung-10 K-split dispatch wiring.

For each -vd shape: run the WIRED awq_gemm_triton(x, q, s, z, 8) (now routes
M==1 to awq_gemv_splitk_triton) vs the rung-9 tile-winner GEMV output
(verified vs dot at rel<=2.4e-4 in rung 9) as reference. Gate rel<0.02 +
flushed timing of both paths. GPU 1 only — GPU 0 untouched until this PASSes.
"""

import torch
from vllm.model_executor.layers.quantization.awq_triton import (
    awq_gemm_triton,
    awq_gemv_kernel,
)

import triton

# (name, N, K, calls, rung-9 winner cfg)
SHAPES = [
    ("gate/up", 9216, 2560, 64, (128, 128, 4, 3)),
    ("down   ", 2560, 9216, 32, (16, 128, 8, 3)),
    ("qkv+q ", 8192, 2560, 32, (128, 64, 4, 3)),
    ("z     ", 4096, 2560, 24, (128, 128, 4, 3)),
    ("out/o ", 2560, 4096, 32, (16, 128, 8, 3)),
    ("k/v   ", 1024, 2560, 16, (16, 128, 8, 3)),
]

FLUSH = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")


def bench(fn, iters=60):
    for _ in range(3):
        fn()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    torch.cuda.synchronize()
    for a, b in ev:
        FLUSH.sum()
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[iters // 2] * 1000


ok = True
tot_old = tot_new = 0.0
for name, N, K, calls, (bn, bk, w, st) in SHAPES:
    torch.manual_seed(42)
    q = torch.randint(-2**31, 2**31 - 1, (K, N // 8), dtype=torch.int32, device="cuda")
    s = torch.randn(K // 128, N, dtype=torch.float16, device="cuda") * 0.01
    z = torch.randint(-2**31, 2**31 - 1, (K // 128, N // 8), dtype=torch.int32, device="cuda")
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")

    # reference: rung-9 tile winner launched directly
    ref = torch.empty(N, dtype=torch.float16, device="cuda")
    awq_gemv_kernel[(triton.cdiv(N, bn),)](
        x, q, ref, z, s, K, N, 128,
        BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=st,
    )
    ref = ref.float()

    new = awq_gemm_triton(x, q, s, z, 8).float()   # wired dispatch -> splitk
    rel = (new - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)

    # rung-11 reuse-hazard check: with persistent partials, a second call with
    # a different input must not corrupt the first call's already-returned
    # tensor, and a repeat of the first call must reproduce it.
    x2 = torch.randn(1, K, dtype=torch.float16, device="cuda")
    o2 = awq_gemm_triton(x2, q, s, z, 8).float()
    o1b = awq_gemm_triton(x, q, s, z, 8).float()
    rel_b = (o1b - new).abs().max().item() / (new.abs().max().item() + 1e-9)
    assert (new - o2).abs().max().item() > 0, "different input gave same output?"
    if rel_b >= 0.02:
        print(f"{name}: REUSE HAZARD rel_b={rel_b:.2e} FAIL", flush=True)
        ok = False

    def old():
        o = torch.empty(N, dtype=torch.float16, device="cuda")
        awq_gemv_kernel[(triton.cdiv(N, bn),)](
            x, q, o, z, s, K, N, 128,
            BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=st,
        )

    t_old = bench(old)
    t_new = bench(lambda: awq_gemm_triton(x, q, s, z, 8))
    tot_old += t_old * calls / 1000
    tot_new += t_new * calls / 1000
    passed = rel < 0.02
    ok &= passed
    print(f"{name} N={N:5d} K={K:5d}: rel={rel:.2e} "
          f"old={t_old:6.1f}us new={t_new:6.1f}us ({t_old - t_new:+6.1f}) "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    del q, s, z, x, x2, ref, new, o2, o1b
    torch.cuda.empty_cache()

print(f"\nINT4 GEMV block: {tot_old:.2f} -> {tot_new:.2f} ms/token "
      f"(saved {tot_old - tot_new:.2f})", flush=True)
print("GATE:", "ALL PASS" if ok else "FAIL - DO NOT PROMOTE", flush=True)
