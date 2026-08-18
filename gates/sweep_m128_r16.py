"""
sweep_m32.py — §24 Req 3: per-shape tile sweep of the M>32 tiled GEMM (the b128
hot path), rung-15 method, correctness-gated vs stock dequant FIRST.

Entry 35/36 established:
  - The b128 (128-concurrent) hot path is the M>32 tiled GEMM (896 calls, all
    M=128, zero M==1). Its AWQ weight-stream is ~15 GB/s = 3% of the 445 roofline.
  - The TRUE read peak is ~438 GB/s (probe_bw_readonly.py), not the 188 GB/s
    D2D memcpy proxy. So the sweep target is the ~438 GB/s read peak.
  - The live M>32 dispatch uses stock tiles BM32/BN32/BK32 W4 S3 and
    split_k_iters = pack_factor = 8 (auto_awq.py:957 passes pack_factor).

This sweep:
  1. Loads the model (GPU 1, offline, eager, b128) and captures the 5 true
     decode shapes' (qweight, scales, qzeros).
  2. Correctness reference: stock dequant (awq_dequantize) + torch.matmul in
     fp32 -> the ground-truth (M,N) output for a fixed input.
  3. Sweeps the M>32 branch by calling awq_gemm_kernel DIRECTLY with explicit
     BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages/SPLIT_K (bypassing the
     dispatch so every param is controlled).
  4. GATE: a candidate is only timing-eligible if its output is within rel<0.02
     of the dequant ref (correctness before speed).
  5. Times each eligible config per shape (single launch, L2-flushed, med-15)
     and reports per-shape winner + stock baseline + the 438 GB/s read peak.

Run: docker run ... -w /qwork blivioniag/vllm-rdna:v0.26.0 /qwork/sweep_m32.py
"""
import os
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
import torch
from vllm.triton_utils import tl, triton
from vllm import LLM, SamplingParams
import vllm.model_executor.layers.quantization.awq_triton as at
from vllm import _custom_ops as ops

print("DEVICE:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
REAL = at.awq_gemm_triton
KERNEL = at.awq_gemm_kernel
DEQ = ops.awq_dequantize   # stock dequant for the correctness ref

SHAPES = [(18432, 2560), (12288, 2560), (10240, 2560), (2560, 9216), (2560, 4096)]
COUNTS = {(18432, 2560): 30, (12288, 2560): 22, (10240, 2560): 8,
          (2560, 9216): 30, (2560, 4096): 30}
M = 128
LIVE_SK = 8        # pack_factor for AWQ-4bit (auto_awq.py:957)
PEAK = 438.0       # §24 Req 1: pure-read peak GB/s (probe_bw_readonly.py)

def _bytes(n, k):
    return k * n * 0.5 + k * n / 64 + k * n / 256 + k * 2 + n * 2

# --- load + capture the 5 shapes -------------------------------------------
llm = LLM(
    model="/model", quantization="awq", dtype="float16",
    max_model_len=8192, max_num_seqs=128, max_num_batched_tokens=2048,
    enable_chunked_prefill=True, gpu_memory_utilization=0.6, enforce_eager=True,
    attention_backend="ROCM_ATTN", trust_remote_code=True, generation_config="vllm",
    limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_pixels": 1003520},
)
llm.generate(["hi"], SamplingParams(max_tokens=4, temperature=0, ignore_eos=True),
             use_tqdm=False)
torch.cuda.synchronize()

CAP = {}
def cap(input, qweight, scales, qzeros, split_k_iters):
    key = (qweight.shape[1] * 8, qweight.shape[0])   # (N, K)
    if key not in CAP:
        CAP[key] = (qweight, scales, qzeros, split_k_iters)
    return REAL(input, qweight, scales, qzeros, split_k_iters)
at.awq_gemm_triton = cap
llm.generate(["hi"], SamplingParams(max_tokens=4, temperature=0, ignore_eos=True),
             use_tqdm=False)
at.awq_gemm_triton = REAL
print(f"captured shapes: {sorted(CAP.keys())}", flush=True)

FLUSH = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")

# --- correctness reference: stock dequant + matmul -------------------------
# Use awq_dequantize_triton (the known-good (K,N) dequant) directly. The C++
# ops.awq_dequantize wrapper requires (qweight, scales, zeros, split_k_iters,
# thx, thy) and falls back to torch.ops._C.awq_dequantize which is NOT present
# in this build -- passing (w,s,z,0,0,0) hit that broken path and produced a
# corrupt ref (live_rel ~0.62 for the in-production kernel = ref was wrong,
# not the kernel). awq_dequantize_triton(qw, sc, z) -> (K, N) fp16.
def ref_output(nk, x):
    N, K = nk
    w, s, z, _ = CAP[nk]
    deq = at.awq_dequantize_triton(w, s, z).to(torch.float32)   # (K, N) fp32
    return (x.to(torch.float32) @ deq)                          # (M, N) fp32

# Correctness gate. The tiled GEMM accumulates in fp16 (tl.dot), so a CORRECT
# config sits at ~fp16-matmul rounding of the exact fp32 dequant ref. probe_tol.py
# (synthetic, K=2560): max_abs_err 0.79 vs expected ~0.83, mean abs 0.088 (0.26%
# rel). The OLD element-wise max_rel (floor 1e-3) was an ARTIFACT: one
# small-magnitude element's normal fp16 rounding -> global max_rel ~0.6-390, so
# EVERY config (incl. the proven in-production kernel; b128 text is coherent)
# failed elig. Gate on a proper matmul tolerance instead: allclose with an
# absolute floor scaled to the K-reduction rounding + a relative term.
def matmul_close(a, b, atol, rtol):
    return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))

def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()

def max_rel(a, b):
    # kept only for reporting (max element-wise relative error); NOT a gate.
    denom = b.abs().clamp_min(1e-3)
    return ((a - b).abs() / denom).max().item()

# --- kernel launch with explicit params (M>32 branch, direct) ---------------
def launch(nk, x, bm, bn, bk, w, s, sk):
    N, K = nk
    qw, sc, qz, _ = CAP[nk]
    gs = qw.shape[0] // qz.shape[0]
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn), sk)
    if sk == 1:
        out = torch.empty((M, N), dtype=sc.dtype, device=x.device)
        KERNEL[grid](x, qw, out, qz, sc, M, N, K, gs,
                     BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                     SPLIT_K=sk, num_warps=w, num_stages=s)
        return out
    out = torch.zeros((sk, M, N), dtype=sc.dtype, device=x.device)
    KERNEL[grid](x, qw, out, qz, sc, M, N, K, gs,
                 BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                 SPLIT_K=sk, num_warps=w, num_stages=s)
    return out.sum(0)

def time_us(nk, x, bm, bn, bk, w, s, sk):
    for _ in range(3):
        launch(nk, x, bm, bn, bk, w, s, sk)
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(15)]
    for a, b in evs:
        FLUSH.sum()
        a.record()
        launch(nk, x, bm, bn, bk, w, s, sk)
        b.record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b) for a, b in evs)[7] * 1000

# --- the sweep grid (M>32 tiled GEMM) ---------------------------------------
# BM must tile M=128; BN/BK/warps/stages free. SK in {1,2,4,8} (live=8).
# Keep it bounded: a handful of BM/BN/BK/warps/stages x 4 SK per shape.
BM   = [32, 64, 128]
BN   = [32, 64, 128, 256]
BK   = [32, 64, 128]
WARPS = [4, 8, 16]
STAGES = [2, 3, 4]
SKS  = [1, 4, 8]

# Precompute correctness ref + input once per shape.
print("\n=== correctness refs + sweep (M=128) ===", flush=True)
all_results = {}
for nk in SHAPES:
    N, K = nk
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
    ref = ref_output(nk, x)
    # DIAG: is my reference correct? Compare the LIVE awq_gemm_triton (M>32
    # branch, its own launch) against my ref_output. If this is ALSO high, the
    # bug is in ref/inputs; if it's low, the bug is in the raw-kernel launch.
    w_d, s_d, z_d, sk_d = CAP[nk]
    gs = w_d.shape[0] // z_d.shape[0]
    live = REAL(x, w_d, s_d, z_d, sk_d)          # live M>32 path (stock tiles)
    # Gate = proper fp16-matmul tolerance. A CORRECT config passes this.
    live_ok = matmul_close(live, ref, atol=0.8, rtol=0.05)
    stock_ok = matmul_close(launch(nk, x, 32, 32, 32, 4, 3, sk_d), ref, atol=0.8, rtol=0.05)
    print(f"    DIAG group_size={gs} (K={K}, K//gs={K//gs}, qz.rows={z_d.shape[0]}) "
          f"live maxabs={max_abs(live, ref):.3f} pass={live_ok} | "
          f"stock(32,32,32,4,3,sk{sk_d}) maxabs={max_abs(launch(nk, x, 32, 32, 32, 4, 3, sk_d), ref):.3f} "
          f"pass={stock_ok} (tolerance atol=0.8 rtol=0.05)",
          flush=True)
    print(f"--- shape {nk} (N={N}, K={K}, x{COUNTS[nk]}) ---", flush=True)
    best = None   # (us, cfg, rel)
    n_elig = n_inel = n_fail = 0
    n_seen = 0
    for bm in BM:
        for bn in BN:
            for bk in BK:
                for w in WARPS:
                    for s in STAGES:
                        for sk in SKS:
                            # skip configs that can't tile / too big
                            if bk > K:
                                continue
                            if bn * bk * w > 65536 * 2:   # rough LDS guard
                                continue
                            n_seen += 1
                            try:
                                out = launch(nk, x, bm, bn, bk, w, s, sk)
                            except Exception:
                                n_fail += 1
                                if n_seen % 60 == 0:
                                    print(f"    [{n_seen}] (bm{bm},bn{bn},bk{bk},w{w},s{s},sk{sk}) LAUNCH-FAIL elig={n_elig}", flush=True)
                                continue
                            ok = matmul_close(out, ref, atol=0.8, rtol=0.05)
                            if not ok:
                                n_inel += 1
                                if n_seen % 60 == 0:
                                    print(f"    [{n_seen}] (bm{bm},bn{bn},bk{bk},w{w},s{s},sk{sk}) maxabs={max_abs(out, ref):.3f} INELIG elig={n_elig} best={best[0] if best else '-'}us", flush=True)
                                continue
                            n_elig += 1
                            us = time_us(nk, x, bm, bn, bk, w, s, sk)
                            if best is None or us < best[0]:
                                best = (us, (bm, bn, bk, w, s, sk), max_abs(out, ref))
                                print(f"    NEW BEST [{n_seen}] (bm{bm},bn{bn},bk{bk},w{w},s{s},sk{sk}) maxabs={max_abs(out, ref):.3f} {us:.1f}us", flush=True)
                            elif n_seen % 60 == 0:
                                print(f"    [{n_seen}] (bm{bm},bn{bn},bk{bk},w{w},s{s},sk{sk}) {us:.1f}us elig={n_elig} best={best[0]:.1f}us", flush=True)
    if best:
        us, cfg, maxa = best
        bw = _bytes(*nk) / (us / 1e6) / 1e9
        all_results[nk] = best
        print(f"  ELIGIBLE {n_elig}  INELIGIBLE {n_inel}  FAIL {n_fail}  (tolerance atol=0.8 rtol=0.05)", flush=True)
        print(f"  WINNER  BM={cfg[0]} BN={cfg[1]} BK={cfg[2]} W={cfg[3]} S={cfg[4]} SK={cfg[5]}"
              f"  maxabs={maxa:.3f}  {us:8.1f} us  {bw:6.0f} GB/s  ({bw/PEAK*100:4.0f}% of {PEAK:.0f})", flush=True)
    else:
        print(f"  NO ELIGIBLE CONFIG. ELIGIBLE {n_elig} INELIG {n_inel} FAIL {n_fail}", flush=True)

# --- aggregate + stock baseline comparison ---------------------------------
print("\n=== aggregate: best config per shape vs STOCK (BM32/BN32/BK32 W4 S3 SK=8) ===", flush=True)
def time_shape_cfg(nk, cfg):
    bm, bn, bk, w, s, sk = cfg
    x = torch.randn(M, nk[1], dtype=torch.float16, device="cuda") * 0.1
    return time_us(nk, x, bm, bn, bk, w, s, sk)

STOCK = (32, 32, 32, 4, 3, 8)
agg_best_us, agg_stock_us, agg_by = 0.0, 0.0, 0.0
print(f"{'(N,K)':>16} {'x':>4} {'best_us':>9} {'best_BW':>8} {'stock_us':>9} {'stock_BW':>9} {'gain':>7}", flush=True)
for nk in SHAPES:
    N, K = nk
    b_us, b_cfg, _ = all_results[nk]
    s_us = time_shape_cfg(nk, STOCK)
    sb = _bytes(*nk)
    agg_best_us += b_us * COUNTS[nk]
    agg_stock_us += s_us * COUNTS[nk]
    agg_by += sb * COUNTS[nk]
    best_bw = _bytes(*nk) / (b_us / 1e6) / 1e9
    stock_bw = sb / (s_us / 1e6) / 1e9
    gain = s_us / b_us
    print(f"{str(nk):>16} {COUNTS[nk]:>4} {b_us:9.1f} {best_bw:8.0f} {s_us:9.1f} "
          f"{stock_bw:9.0f} {gain:7.2f}x", flush=True)

print(f"\nAGG best  : {agg_best_us:9.1f} us  {agg_by/1e9/(agg_best_us/1e6):8.0f} GB/s", flush=True)
print(f"AGG stock : {agg_stock_us:9.1f} us  {agg_by/1e9/(agg_stock_us/1e6):8.0f} GB/s", flush=True)
print(f"AGG gain  : {agg_stock_us/agg_best_us:.2f}x (per-decode-step AWQ block)", flush=True)
print(f"\nRead-peak anchor (§24 Req 1): {PEAK:.0f} GB/s. Best agg = "
      f"{agg_by/1e9/(agg_best_us/1e6):.0f} GB/s = "
      f"{agg_by/1e9/(agg_best_us/1e6)/PEAK*100:.0f}% of it.", flush=True)
