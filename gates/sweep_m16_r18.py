import math
import os
import sys
import time
import torch

# Breakable-cudagraph capture does OOM under enforce_eager; disable it so the
# in-process LLM() init takes the eager path cleanly.
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
sys.path.insert(0, "/qwork")

import vllm.model_executor.layers.quantization.awq_triton as at

REAL = at.awq_gemm_triton          # host-patched live path (the reference under test)
KERNEL = at.awq_gemm_kernel        # the triton kernel both bands launch
torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"

# -vd decode shape set (Entry 36). (N,K) and count per token.
SHAPES = [(18432, 2560), (12288, 2560), (10240, 2560), (2560, 9216), (2560, 4096)]
COUNTS = {(18432, 2560): 30, (12288, 2560): 22, (10240, 2560): 8,
          (2560, 9216): 30, (2560, 4096): 30}
MS = [8, 16, 32]
PEAK = 438.0  # GB/s (Entry 36 read-only bandwidth ceiling)

# Stock (current) M<=32 config: BM16 BN128 BK64 W8 S3, SK = 1 if K<=4096 else 8.
def stock_cfg(n, k):
    return (16, 128, 64, 8, 3, (1 if k <= 4096 else 8))

def _bytes(n, k):
    return k * n * 0.5 + k * n / 64 + k * n / 256 + k * 2 + n * 2


def init_llm():
    from vllm import LLM
    return LLM(model="/model", quantization="awq", dtype="float16",
               max_model_len=8192, max_num_seqs=16, max_num_batched_tokens=512,
               gpu_memory_utilization=0.50, enforce_eager=True,
               attention_backend="ROCM_ATTN", trust_remote_code=True,
               generation_config="vllm",
               limit_mm_per_prompt={"image": 1},
               mm_processor_kwargs={"max_pixels": 1003520})


CAP = {}
def cap(x, qweight, scales, qzeros, split_k_iters, *a, **k):
    # Real signature: awq_gemm_triton(input, qweight, scales, qzeros,
    # split_k_iters, block_size_m=32, block_size_n=32, block_size_k=32).
    # From the asserts: qweight.shape == (K, N//8)  (K is FULL, not packed;
    # only the N axis is int4-packed). So N = qweight.shape[1]*8, K = qweight.shape[0].
    N = qweight.shape[1] * 8
    K = qweight.shape[0]
    group_size = qweight.shape[0] // qzeros.shape[0]
    CAP[(N, K)] = (x, qweight, qzeros, scales, group_size)
    return REAL(x, qweight, scales, qzeros, split_k_iters)

def capture(llm):
    from vllm import SamplingParams
    SP = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    at.awq_gemm_triton = cap
    CAP.clear()
    llm.generate(["hi"], SP, use_tqdm=False)
    for _ in range(2):
        llm.generate(["hi"], SP, use_tqdm=False)
    at.awq_gemm_triton = REAL
    print("captured shapes:", sorted(CAP.keys()), flush=True)
    return CAP

# ---------------------------------------------------------------------------
FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
def time_us(fn, iters=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        FLUSH.zero_()
        torch.cuda.synchronize()
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1) * 1000.0)
    ts.sort()
    return ts[len(ts)//2]

def cdiv(a, b):
    return (a + b - 1) // b

def launch(m, n, k, x, qweight, qzeros, scales, group_size, bm, bn, bk, w, s, sk):
    grid = (cdiv(m, bm) * cdiv(n, bn), sk)
    if sk == 1:
        result = torch.empty((m, n), device=DEV, dtype=x.dtype)
    else:
        result = torch.zeros((sk, m, n), device=DEV, dtype=x.dtype)
    KERNEL[grid](x, qweight, result, qzeros, scales, m, n, k, group_size,
                 bm, bn, bk, sk, num_warps=w, num_stages=s)
    if sk > 1:
        result = result.sum(0)
    return result

def matmul_close(a, b):
    return torch.allclose(a.float(), b.float(), atol=0.8, rtol=0.05)

def ref_for(m, n, k, x, qweight, qzeros, scales, group_size):
    # awq_dequantize_triton(qweight, scales, zeros, block_size_x=32, block_size_y=32)
    # returns the dequantized weight of shape (K, N) == (qweight.shape[0], qweight.shape[1]*8).
    deq = at.awq_dequantize_triton(qweight, scales, qzeros, 32, 32).to(torch.float32)
    return x.float() @ deq

# The capture hook stores the activation `x` last seen for each (N,K) shape
# during 3x generate(["hi"]). That `x` is a PREFILL activation (M = seq_len),
# and it can carry non-finite values (attention over a short/dummy context
# produces Inf/NaN in the residual stream). The numerics gate compares the
# GEMM output to a float32 ref, so the input MUST be finite or both live and
# ref are NaN (this was the v5-v9 blocker: x16_fin=False -> everything NaN).
#
# Fix: build a (32, K) buffer, copy the captured row in, then replace any
# non-finite element with a small finite sentinel. The timing is dominated by
# the N*K int4 weight stream, so the exact activation values (finite or not)
# do not move the timing; only finiteness matters for the gate.
def pad_x(x, k):
    buf = torch.zeros(32, k, device=DEV, dtype=x.dtype)
    buf[: x.shape[0]].copy_(x)
    # Sanitize: replace non-finite with a finite value so the matmul is well-
    # defined. Use the finite elements' scale if any exist, else 0.1.
    xf = x.float()
    fin_mask = torch.isfinite(xf)
    if fin_mask.any():
        sentinel = xf[fin_mask].abs().clamp(min=1e-3, max=4.0).mean().to(buf.dtype)
    else:
        sentinel = buf.new_tensor(0.1)
    buf[~torch.isfinite(buf.float())] = sentinel
    return buf

# ---------------------------------------------------------------------------
# Prior-anchored grid.
#
# Rung-16 M>32 winners are (BM64,BN128,BK32,W8,S2,SK). The two levers that
# differ from stock (BK64,W8,S3) are BK=32 and S=2 — both favour fewer/cheaper
# LDS stages. For M<=32 BM drops to 16 (or 8 for M=8). So we sweep the axes
# that actually move the needle — BM, SK, and BK/S — and hold BN=128, W=8 as
# the rung-16-validated defaults, with a small W/S perturbation. That keeps the
# grid to ~40 configs/shape-M (JIT-cheap) instead of 378.
# ---------------------------------------------------------------------------
BM_BY_M = {8: [8, 16], 16: [16, 32], 32: [16, 32]}
BN_FIX = [128]
BK_LIST = [32, 64]
W_LIST = [8]
S_LIST = [2, 3]
SK_LIST = [1, 2, 4, 8]

def feasible(m, n, k, bm, bn, bk, w, s, sk):
    if bk > k:           return False
    if sk * bk > k:      return False
    if bk * m * 2 > 65536: return False
    lds = (bm * bk + bn * bk) * 2 * s
    if lds > 128 * 1024: return False
    return True

def sweep_configs(m, n, k):
    out = []
    for bm in BM_BY_M[m]:
        for bn in BN_FIX:
            for bk in BK_LIST:
                for w in W_LIST:
                    for s in S_LIST:
                        for sk in SK_LIST:
                            if feasible(m, n, k, bm, bn, bk, w, s, sk):
                                out.append((bm, bn, bk, w, s, sk))
    # de-dup preserving order
    seen = set(); uniq = []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq

def main():
    print("=== Rung-18 M<=32 prior-anchored sweep (in-process, real weights) ===", flush=True)
    llm = init_llm()
    CAP = capture(llm)
    for s in SHAPES:
        assert s in CAP, f"missing shape {s} in capture"

    # DIAG: confirm live M<=32 path matches ref at M=16 (validates the gate).
    # REAL = awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters).
    print("=== DIAG: live M<=32 path vs ref (M=16) ===", flush=True)
    print(f"  REAL is awq_gemm_triton: {REAL is at.awq_gemm_triton}", flush=True)
    print(f"  at.awq_gemm_triton is cap: {at.awq_gemm_triton is cap}", flush=True)
    for (n, k) in SHAPES:
        x, qweight, qzeros, scales, group_size = CAP[(n, k)]
        x32 = pad_x(x, k)
        x16 = x32[:16].contiguous()
        x_fin = torch.isfinite(x16.float()).all().item()
        ref = ref_for(16, n, k, x16, qweight, qzeros, scales, group_size)
        ref_fin = torch.isfinite(ref.float()).all().item()
        live = REAL(x16, qweight, scales, qzeros, 1)  # split_k_iters=1 for the ref path
        live_fin = torch.isfinite(live.float()).all().item()
        la = (live.float() - ref).abs().max().item()
        print(f"  ({n}, {k}) M=16 x16_fin={x_fin} ref_fin={ref_fin} live_fin={live_fin} "
              f"maxabs={la:.3f} pass={matmul_close(live, ref)}", flush=True)

    results = {m: {} for m in MS}

    for (n, k) in SHAPES:
        x, qweight, qzeros, scales, group_size = CAP[(n, k)]
        x32 = pad_x(x, k)
        cnt = COUNTS[(n, k)]
        b = _bytes(n, k)
        print(f"\n===== shape N={n} K={k} count={cnt} ({b/1e6:.1f} MB) =====", flush=True)
        for m in MS:
            x_m = x32[:m].contiguous()
            ref = ref_for(m, n, k, x_m, qweight, qzeros, scales, group_size)
            cfgs = sweep_configs(m, n, k)
            best = None
            for cfg in cfgs:
                fn = lambda: launch(m, n, k, x_m, qweight, qzeros, scales, group_size, *cfg)
                t = time_us(fn)
                if best is None or t < best[1]:
                    best = (cfg, t)
            bm, bn, bk, w, s, sk = best[0]
            out = launch(m, n, k, x_m, qweight, qzeros, scales, group_size, bm, bn, bk, w, s, sk)
            ok = matmul_close(out, ref)
            stock = stock_cfg(n, k)
            t_stock = time_us(lambda: launch(m, n, k, x_m, qweight, qzeros, scales, group_size, *stock))
            gbs = b / 1e3 / best[1]
            print(f"  WIN M={m} -> ({bm},{bn},{bk},w{w},s{s},SK{sk}) "
                  f"t={best[1]:.2f}us GB/s={gbs:.0f} stock={stock} t_stock={t_stock:.2f}us "
                  f"gain={t_stock/best[1]:.2f}x n_cfg={len(cfgs)} pass={ok}", flush=True)
            results[m][(n, k)] = {"cfg": (bm, bn, bk, w, s, sk), "t_us": best[1],
                                   "gbps": gbs, "ok": ok}

    # ---- Aggregate per M ----
    print("\n=== Aggregate per M (all 5 shapes x counts) ===", flush=True)
    for m in MS:
        tot = sum(results[m][s]["t_us"] * COUNTS[s] for s in SHAPES)
        totstock = 0.0
        for s in SHAPES:
            n, k = s
            x, qweight, qzeros, scales, group_size = CAP[s]
            x_m = pad_x(x, k)[:m].contiguous()
            st = stock_cfg(n, k)
            totstock += time_us(lambda: launch(m, n, k, x_m, qweight, qzeros, scales, group_size, *st)) * COUNTS[s]
        okall = all(results[m][s]["ok"] for s in SHAPES)
        print(f"M={m}: agg={tot/1000:.3f} ms/token  stock={totstock/1000:.3f} ms/token  "
              f"gain={totstock/tot:.2f}x  all_ok={okall}", flush=True)

    print("\n=== Winner table (BM,BN,BK,W,S,SK) per (shape, M) ===", flush=True)
    for s in SHAPES:
        row = "  " + str(s) + ":"
        for m in MS:
            c = results[m][s]["cfg"]
            row += f"  M{m}=({c[0]},{c[1]},{c[2]},w{c[3]},s{c[4]},SK{c[5]})"
        print(row, flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
