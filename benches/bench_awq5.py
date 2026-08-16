"""§22 item-2 step 2: K-split GEMV sweep for the latency-bound -vd shapes.

Rung-9 fact: small-N shapes run 8-27% of peak BW; cdiv(N,BN) programs can't fill
~80 CUs. awq_gemv_splitk_triton adds a K-split grid axis + fp32 partials + tiny
reduce kernel. This sweep finds per-shape (BN, BK, SPLIT, W) vs the LIVE rung-9
dispatch (awq_gemm_triton M==1 = exactly what serves today).

Run: docker cp awq_triton.py into quant-run first, then
     docker exec quant-run python3 -u /qwork/bench_awq5.py   (GPU 1)
"""

import torch
from vllm.model_executor.layers.quantization.awq_triton import (
    awq_gemm_triton,
    awq_gemv_splitk_triton,
)

# (name, N, K, calls/token) — all six -vd shapes (big-N ones included: more
# parallelism may help them too)
SHAPES = [
    ("gate/up", 9216, 2560, 64),
    ("down   ", 2560, 9216, 32),
    ("qkv+q ", 8192, 2560, 32),
    ("z     ", 4096, 2560, 24),
    ("out/o ", 2560, 4096, 32),
    ("k/v   ", 1024, 2560, 16),
]

BW = 445e9
FLUSH = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")

GRID = [
    (bn, bk, sp, w)
    for sp in (2, 4, 8, 16)
    for bn in (16, 32, 64)
    for bk in (32, 64, 128)
    for w in (2, 4, 8)
]


def bench(fn, iters=60):
    for _ in range(3):
        fn()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    torch.cuda.synchronize()
    for a, b in ev:
        FLUSH.sum()                      # evict L2 between iterations
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[iters // 2] * 1000          # median, us


cur_ms = 0.0
best_ms = 0.0
CUR_MS = 18.62   # 1000/53.7 standing decode budget

for name, N, K, calls in SHAPES:
    torch.manual_seed(42)
    q = torch.randint(-2**31, 2**31 - 1, (K, N // 8), dtype=torch.int32, device="cuda")
    s = torch.randn(K // 128, N, dtype=torch.float16, device="cuda") * 0.01
    z = torch.randint(-2**31, 2**31 - 1, (K // 128, N // 8), dtype=torch.int32, device="cuda")
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")

    ref = awq_gemm_triton(x, q, s, z, 8).float()   # live rung-9 dispatch

    wbytes = K * N * 0.5 + K * N / 64 + K * N / 256
    floor = wbytes / BW * 1e6

    serve_us = bench(lambda: awq_gemm_triton(x, q, s, z, 8))

    print(f"--- {name} N={N:5d} K={K:5d} x{calls:<3d} (floor {floor:6.1f}us) ---", flush=True)
    print(f"    SERVING (rung-9 table): {serve_us:7.1f} us ({floor/serve_us*100:4.0f}% peak)", flush=True)

    results = []
    for bn, bk, sp, w in GRID:
        if K % sp or (K // sp) % bk:
            continue
        try:
            o = awq_gemv_splitk_triton(x, q, s, z, bn, bk, sp, w, 3).float()
            rel = (o - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            if rel >= 0.02:
                continue
            us = bench(lambda: awq_gemv_splitk_triton(x, q, s, z, bn, bk, sp, w, 3))
            results.append((us, bn, bk, sp, w))
        except Exception:
            continue
    results.sort()
    for us, bn, bk, sp, w in results[:3]:
        print(f"    splitk  BN={bn:3d} BK={bk:3d} SP={sp:2d} W={w}: {us:7.1f} us "
              f"({floor/us*100:4.0f}% peak)", flush=True)
    if results:
        best_us = results[0][0]
        cur_ms += serve_us * calls / 1000
        best_ms += min(serve_us, best_us) * calls / 1000
        print(f"    -> delta {serve_us - best_us:+7.1f} us x{calls} = "
              f"{(serve_us - best_us) * calls / 1000:+6.3f} ms/token", flush=True)
    del q, s, z, x, ref
    torch.cuda.empty_cache()

saved = cur_ms - best_ms
print("\n=== projection ===", flush=True)
print(f"INT4 GEMV block: {cur_ms:.2f} -> {best_ms:.2f} ms/token (saved {saved:.2f})", flush=True)
print(f"projected decode: {CUR_MS:.2f} -> {CUR_MS - saved:.2f} ms/token "
      f"= {1000 / (CUR_MS - saved):.1f} TPS (from 53.7)", flush=True)
