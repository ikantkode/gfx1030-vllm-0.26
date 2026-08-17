"""Split-K intrinsic-vs-extrinsic probe, ROUND 2 (supervisor, 2026-08-17).

Round 1 (probe_sk.py, stock-kernel accident) established the TRUE decode
call set: 120 M==1 awq_gemm calls/token over 5 shapes — (18432,2560)x30,
(12288,2560)x22, (10240,2560)x8, (2560,9216)x30, (2560,4096)x30 — 1.74 GB,
floor 3.91 ms @445 GB/s. The rung-9 shape table (6 shapes / 200 calls) was
stale safetensors-header reading; the rung-10 splitk config 64/32/16/W2 was
swept on shapes that don't exist in decode.

Round 2 (patched awq_triton.py MOUNTED, as live): replay the exact live
sequence through the real splitk dispatch —
ALONE. The full-sequence weight stream (~1.9 GB) self-flushes the 128 MB L2
between reps, so timing is cold/live-representative. Three measurements:
  1. torch.profiler device-time sum per rep (same metric as Entry 26)
  2. hip-event span per rep (shows GPU gaps if CPU launch rate starves)
  3. CUDA-graph replay of the whole sequence (live mechanism, zero CPU
     launches) — the cleanest isolated number.

Verdict rule: isolated ~6.0 ms -> intrinsic; ~4.5 ms -> extrinsic.
Also settles the Entry-26 "128 calls" vs rung-9 "200 calls" discrepancy.
"""
import os

os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

import torch
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType
from vllm import LLM, SamplingParams

print("DEVICE:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)

import vllm.model_executor.layers.quantization.awq_triton as at

REAL = at.awq_gemm_triton
LOG = []


def rec(input, qweight, scales, qzeros, split_k_iters):
    if input.shape[0] == 1:  # decode calls only (prefill has M>1)
        LOG.append((input, qweight, scales, qzeros, split_k_iters))
    return REAL(input, qweight, scales, qzeros, split_k_iters)


llm = LLM(
    model="/hostq/Qwen3.5-4B-AWQ-vd", quantization="awq", dtype="float16",
    max_model_len=8192, max_num_seqs=2, max_num_batched_tokens=2048,
    enable_chunked_prefill=True, gpu_memory_utilization=0.6, enforce_eager=True,
    attention_backend="ROCM_ATTN", trust_remote_code=True, generation_config="vllm",
    limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_pixels": 1003520},
)
SP = SamplingParams(max_tokens=16, temperature=0, ignore_eos=True)

at.awq_gemm_triton = rec  # _custom_ops.awq_gemm re-imports per call -> hook works
llm.generate(["Write a long detailed essay about the history of computing."],
             SP, use_tqdm=False)          # warmup: JIT + partials cache
LOG.clear()
llm.generate(["Explain how bridges stay up."], SP, use_tqdm=False)
assert LOG, "no M==1 calls captured"
at.awq_gemm_triton = REAL

NTOK = 16
assert len(LOG) % NTOK == 0, f"{len(LOG)} calls not divisible by {NTOK}"
per_tok = len(LOG) // NTOK
SEQ = LOG[-per_tok:]  # one token's exact live call sequence, in order
print(f"\ncalls/token: {per_tok}  (Entry-26 said 128, rung-9 sweep said 200)",
      flush=True)

from collections import Counter

cnt = Counter()
bytes_tot = 0
for x, q, s, z, _ in SEQ:
    K, N8 = q.shape
    N = N8 * 8
    cnt[(N, K)] += 1
    bytes_tot += K * N * 0.5 + K * N / 64 + K * N / 256 + K * 2 + N * 2
print(f"unique shapes: {len(cnt)}; bytes/token moved: {bytes_tot/1e9:.2f} GB",
      flush=True)
for (N, K), c in sorted(cnt.items(), key=lambda kv: -kv[1]):
    print(f"  N={N:5d} K={K:5d} x{c}", flush=True)


def replay_seq():
    for x, q, s, z, sk in SEQ:
        REAL(x, q, s, z, sk)


for _ in range(3):
    replay_seq()
torch.cuda.synchronize()

# --- 1. profiler device-time sum (Entry-26 metric) ---
reps = 5
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(reps):
        replay_seq()
    torch.cuda.synchronize()
per_kernel = {}
for e in p.key_averages():
    if e.device_type == DeviceType.CUDA and e.device_time_total > 0:
        per_kernel[e.key] = (e.device_time_total, e.count)
iso_us = sum(v[0] for v in per_kernel.values()) / reps
nk = sum(v[1] for v in per_kernel.values()) / reps
print(f"\n=== ISOLATED, profiler device-sum: {iso_us/1000:.2f} ms/token "
      f"over {nk:.0f} kernels ===", flush=True)
for k, (us, c) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0]):
    print(f"  {k[:60]:60s} {us/reps/1000:7.3f} ms  x{c//reps}", flush=True)

# --- 2. hip-event span (GPU wall incl. gaps; CPU launch rate shows here) ---
evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(20)]
torch.cuda.synchronize()
for a, b in evs:
    a.record()
    replay_seq()
    b.record()
torch.cuda.synchronize()
spans = sorted(a.elapsed_time(b) for a, b in evs)
span_ms = spans[len(spans) // 2]
print(f"\nevent-span median: {span_ms:.2f} ms/token "
      f"(gap vs device-sum: {span_ms - iso_us/1000:+.2f} ms "
      f"= launch-rate/starvation)", flush=True)

# --- 3. CUDA-graph replay (live mechanism; zero CPU launches) ---
try:
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        replay_seq()  # side-stream warmup for capture
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        replay_seq()
    gevs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(20)]
    g.replay()
    torch.cuda.synchronize()
    for a, b in gevs:
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    gs = sorted(a.elapsed_time(b) for a, b in gevs)
    print(f"graph-replay median: {gs[len(gs)//2]:.2f} ms/token", flush=True)
except Exception as ex:
    print(f"graph replay failed ({type(ex).__name__}: {ex}) — "
          "event-span stands", flush=True)

# --- per-shape cold table (L2 flushed) ---
FLUSH = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")
print("\n=== per-shape isolated cold (256MB L2 flush between iters) ===", flush=True)
print(f"{'N':>6} {'K':>6} {'x/tok':>6} {'us':>8} {'MB':>8} {'GB/s':>7} "
      f"{'floor@445':>10} {'%peak':>6}", flush=True)
for (N, K), c in sorted(cnt.items(), key=lambda kv: -kv[1]):
    x, q, s, z, sk = next(t for t in SEQ if t[1].shape[0] == K and t[1].shape[1] * 8 == N)
    fevs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(30)]
    torch.cuda.synchronize()
    for a, b in fevs:
        FLUSH.sum()
        a.record()
        REAL(x, q, s, z, sk)
        b.record()
    torch.cuda.synchronize()
    us = sorted(a.elapsed_time(b) for a, b in fevs)[15] * 1000
    mb = (K * N * 0.5 + K * N / 64 + K * N / 256 + K * 2 + N * 2) / 1e6
    gbs = mb / 1e3 / (us / 1e6)
    floor = mb * 1e12 / 445e9  # us at 445 GB/s
    print(f"{N:6d} {K:6d} {c:6d} {us:8.1f} {mb:8.2f} {gbs:7.0f} "
          f"{floor:10.1f} {gbs/445*100:6.0f}", flush=True)

print(f"\nVERDICT INPUTS: isolated {iso_us/1000:.2f} ms vs live 6.21 ms vs "
      f"floor {bytes_tot/445e9*1000:.2f} ms (@445 GB/s); "
      f"overall {bytes_tot/1e9/(iso_us/1e6):.0f} GB/s isolated "
      f"(lm_head anchor: 439)", flush=True)

# --- config re-sweep on the TRUE shapes (rung-9/10 swept a stale set) ---
from vllm.model_executor.layers.quantization.awq_triton import (
    awq_gemv_splitk_triton as SK,
)

CONFIGS = [
    (bn, bk, sp, w)
    for bn in (16, 32, 64, 128)
    for bk in (32, 64, 128)
    for sp in (2, 4, 8, 16, 32)
    for w in (2, 4, 8)
]
print("\n=== splitk config sweep on true shapes (cold, med-30, "
      "live cfg 64/32/16/W2 in grid) ===", flush=True)
proj_live = proj_best = 0.0
for (N, K), c in sorted(cnt.items(), key=lambda kv: -kv[1]):
    x, q, s, z, sk_ = next(
        t for t in SEQ if t[1].shape[0] == K and t[1].shape[1] * 8 == N)
    gs = q.shape[0] // z.shape[0]
    ref = REAL(x, q, s, z, sk_).float()
    rows = []
    for bn, bk, sp, w in CONFIGS:
        if K % sp or (K // sp) % bk or gs % bk:
            continue
        try:
            o = SK(x, q, s, z, bn, bk, sp, w, 3).float()
            rel = (o - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            if rel >= 0.02:
                continue
            fevs = [(torch.cuda.Event(True), torch.cuda.Event(True))
                    for _ in range(30)]
            torch.cuda.synchronize()
            for a, b in fevs:
                FLUSH.sum()
                a.record()
                SK(x, q, s, z, bn, bk, sp, w, 3)
                b.record()
            torch.cuda.synchronize()
            rows.append((sorted(a2.elapsed_time(b2) for a2, b2 in fevs)[15] * 1000,
                         bn, bk, sp, w))
        except Exception:
            continue
    rows.sort()
    live_row = next((r for r in rows if r[1:] == (64, 32, 16, 2)), None)
    lu = live_row[0] if live_row else float("nan")
    print(f"--- N={N} K={K} x{c} gs={gs} : live-cfg {lu:.1f} us ---", flush=True)
    for us, bn, bk, sp, w in rows[:3]:
        tag = " <-live" if (bn, bk, sp, w) == (64, 32, 16, 2) else ""
        print(f"    BN={bn:3d} BK={bk:3d} SP={sp:2d} W={w}: {us:7.1f} us{tag}",
              flush=True)
    if rows:
        best = rows[0][0]
        proj_live += lu * c / 1000
        proj_best += min(lu, best) * c / 1000
        print(f"    -> {lu - best:+7.1f} us x{c} = "
              f"{(lu - best) * c / 1000:+6.3f} ms/token", flush=True)

saved = proj_live - proj_best
print(f"\nSWEEP PROJECTION: splitk block {proj_live:.2f} -> {proj_best:.2f} "
      f"ms/token (saved {saved:.2f}); decode 12.64 -> "
      f"{12.64 - saved:.2f} ms = {1000 / (12.64 - saved):.1f} TPS", flush=True)
