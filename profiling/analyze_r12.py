"""Analyze the rung-12 capture: sorted kernel long-poles + elementwise chains
+ inter-kernel gaps, normalized per decode token (255 decode steps in the
profiled 256-token generate: 1 prefill forward + 255 decodes)."""

import csv
import json
import sys

CSV = "/qwork/prof_r12_kernels.csv"
TRACE = "/qwork/prof_r12_trace.json"
NTOK = 255.0  # decode steps

rows = list(csv.DictReader(open(CSV)))
for r in rows:
    r["us"] = float(r["cuda_total_us"])
kern = [r for r in rows if r["us"] > 0]

print("=" * 100)
print("TOP 30 KERNELS BY GPU TIME (total ms | ms/token | count | avg us)")
print("=" * 100)
for r in sorted(kern, key=lambda r: -r["us"])[:30]:
    print(f"{r['us']/1000:8.3f} ms | {r['us']/1000/NTOK:7.4f} ms/tok | {int(r['count']):7d} x "
          f"{r['us']/int(r['count']):7.2f} us | {r['name'][:70]}")

GROUPS = [
    ("INT4 splitk gemv ", ("awq_gemv_splitk_kernel",)),
    ("INT4 splitk reduce", ("awq_gemv_splitk_reduce",)),
    ("INT4 gemv (old) ", ("awq_gemv_kernel",)),
    ("LLMM1/fp16 gemv ", ("LLMM1", "Lazy", "gemv", "Gemv", "GEMV")),
    ("rocBLAS/gemm    ", ("Cijk", "rocblas", "gemm", "Gemm")),
    ("attention       ", ("attn", "Attn", "attention", "Attention")),
    ("FLA/linear attn ", ("fused_recurrent", "fla", "chunk_", "mamba", "causal_conv", "sgconv")),
    ("elementwise/cast", ("elementwise", "ElementWise", "cast", "Cast", "copy_", "Copy",
                          "cat", "Cat", "Memcpy", "fill", "Fill", "index", "Index",
                          "triu", "where", "silu", "Silu", "direct_copy", "vectorized")),
    ("reduce/sum     ", ("reduce", "Reduce", "sum", "argmax", "softmax")),
]
print("=" * 100)
print("GROUPED (ms/token, over 255 decode steps)  [overlap possible: substring match]")
print("=" * 100)
groups = {name: 0.0 for name, _ in GROUPS}
for r in kern:
    for name, keys in GROUPS:
        if any(k in r["name"] for k in keys):
            groups[name] += r["us"]
            break
for name, _ in GROUPS:
    print(f"{groups[name]/1000/NTOK:8.4f} ms/tok | {name}")
ungrouped = sum(r["us"] for r in kern) - sum(groups.values())
print(f"{ungrouped/1000/NTOK:8.4f} ms/tok | (everything else)")
print(f"{sum(r['us'] for r in kern)/1000/NTOK:8.4f} ms/tok | TOTAL kernel GPU time")

# ---- trace gap analysis ----
print("=" * 100)
print("TRACE: inter-kernel gaps (steady window)")
print("=" * 100)
with open(TRACE) as f:
    tr = json.load(f)
ev = [e for e in tr["traceEvents"]
      if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
ev.sort(key=lambda e: e["ts"])
t0, t1 = ev[0]["ts"], ev[-1]["ts"] + ev[-1]["dur"]
print(f"events={len(ev)}  window={(t1-t0)/1e6:.3f} s")

# drop the prefill burst: first 150 ms of the window
ss = [e for e in ev if e["ts"] > t0 + 150_000]
s0, s1 = ss[0]["ts"], ss[-1]["ts"] + ss[-1]["dur"]
span_ms = (s1 - s0) / 1e3
print(f"steady window: {(s1-s0)/1e6:.3f} s, {len(ss)} events, "
      f"{sum(e['dur'] for e in ss)/1e3:.1f} ms busy -> "
      f"{sum(e['dur'] for e in ss)/(s1-s0)*100:.1f}% GPU busy, "
      f"{span_ms/NTOK:.3f} ms/token wall")

streams = {}
for e in ss:
    streams.setdefault(e["tid"], []).append(e)
allgaps = []
for tid, es in streams.items():
    for a, b in zip(es, es[1:]):
        g = b["ts"] - (a["ts"] + a["dur"])
        if g > 0:
            allgaps.append((g, tid, a["name"], b["name"], a["ts"] + a["dur"]))
allgaps.sort(reverse=True)
tot_gap = sum(g for g, *_ in allgaps)
big = [g for g in allgaps if g > 300]
print(f"\ntotal gap time {tot_gap/1e3:.1f} ms over {(s1-s0)/1e6:.3f} s "
      f"({tot_gap/(s1-s0)*100:.1f}% idle); gaps >300us: {len(big)} "
      f"totalling {sum(big)/1e3:.1f} ms")
print("\nTOP 15 GAPS (>300us emphasized):")
for g, tid, a, b, ts in allgaps[:15]:
    print(f"  {g/1e3:8.3f} ms  stream={tid}  after [{a[:48]}] before [{b[:48]}]")

# contiguous elementwise chains: runs of >=4 tiny (<60us) kernels with <5us gaps
CH_US, N_RUN = 60, 4
runs, cur = [], [ss[0]]
for a, b in zip(ss, ss[1:]):
    if b["ts"] - (a["ts"] + a["dur"]) < 5 and a["dur"] < CH_US * 1000 and b["dur"] < CH_US * 1000:
        cur.append(b)
    else:
        if len(cur) >= N_RUN:
            runs.append(cur)
        cur = [b]
if len(cur) >= N_RUN:
    runs.append(cur)
from collections import Counter
run_names = Counter()
for r in runs:
    for e in r:
        run_names[e["name"]] += 1
print(f"\nELEMENTWISE/TINY CHAINS: {len(runs)} runs of >= {N_RUN} kernels <{CH_US}us "
      f"(gap<5us), {sum(len(r) for r in runs)} kernels total, "
      f"{sum(e['dur'] for r in runs for e in r)/1e3:.1f} ms busy")
print("most common kernels inside chains:")
for n, c in run_names.most_common(12):
    print(f"  {c:6d} x {n[:80]}")
