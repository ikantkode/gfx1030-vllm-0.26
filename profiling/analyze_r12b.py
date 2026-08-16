"""Decode-only, device-kernel-only analysis of prof_r12_trace.json.

key_averages() mixes cpu_op/runtime spans with kernels and the prefill
burst; this works straight off cat=="kernel" events:
- decode region = after the END of the last awq_gemm_kernel (prefill's INT4
  GEMM path; decode uses the splitk path, so the splitk count check
  128/step * 255 validates the split).
- lm_head separated from small fp16 GEMVs by duration (LLGemm1 ~2.8ms vs
  in_proj_a/b tens of us).
Normalizes by 255 decode steps. Kernel DURATIONS are honest even though the
profiler inflates wall time (~60us/launch tax) — gaps here are an upper
bound on live gaps.
"""

import json
from collections import defaultdict

TRACE = "/qwork/prof_r12_trace.json"
NTOK = 255

with open(TRACE) as f:
    tr = json.load(f)
ev = [e for e in tr["traceEvents"]
      if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
ev.sort(key=lambda e: e["ts"])
print(f"device events: {len(ev)}")

# decode region starts after prefill's last awq_gemm_kernel
last_prefill = max((e["ts"] + e["dur"] for e in ev if "awq_gemm_kernel" in e["name"]),
                   default=ev[0]["ts"])
dec = [e for e in ev if e["ts"] > last_prefill + 1000]
nsk = sum(1 for e in dec if "awq_gemv_splitk_kernel" in e["name"])
print(f"decode events: {len(dec)}  splitk count {nsk} (expect {128 * NTOK}) "
      f"-> {nsk / NTOK:.1f}/token")

stats = defaultdict(lambda: [0, 0.0, []])
for e in dec:
    s = stats[e["name"]]
    s[0] += 1
    s[1] += e["dur"]
    s[2].append(e["dur"])

print("=" * 110)
print(f"TOP 25 DEVICE KERNELS IN DECODE (ms/token | count/tok | avg us | name)")
print("=" * 110)
for name, (c, tot, durs) in sorted(stats.items(), key=lambda kv: -kv[1][1])[:25]:
    durs.sort()
    p50 = durs[len(durs) // 2]
    print(f"{tot / 1000 / NTOK:8.4f} ms/tok | {c / NTOK:7.2f}/tok | avg {tot / c:7.2f} us "
          f"p50 {p50:7.2f} us | {name[:58]}")

# lm_head vs small fp16 gemv split inside LLGemm1
ll = [e for e in dec if "LLGemm1_kernel" in e["name"]]
big = [e for e in ll if e["dur"] > 1000]
small = [e for e in ll if e["dur"] <= 1000]
print("\nLLGemm1 split (lm_head should be the >1ms population):")
if big:
    print(f"  >1ms : {len(big)} calls, avg {sum(e['dur'] for e in big) / len(big) / 1000:.3f} ms "
          f"-> {sum(e['dur'] for e in big) / 1000 / NTOK:.4f} ms/token")
if small:
    print(f"  <=1ms: {len(small)} calls, avg {sum(e['dur'] for e in small) / len(small):.2f} us "
          f"-> {sum(e['dur'] for e in small) / 1000 / NTOK:.4f} ms/token")

GROUPS = [
    ("INT4 splitk gemv  ", ("awq_gemv_splitk_kernel",)),
    ("INT4 splitk reduce", ("awq_gemv_splitk_reduce",)),
    ("fp16 GEMV >1ms (lm_head) [replaced below]", ()),
    ("fp16 GEMV small   ", ("LLGemm1_kernel",)),
    ("attention         ", ("paged_attention", "attn", "Attention", "flash")),
    ("FLA/linear attn   ", ("fused_recurrent", "causal_conv1d", "chunk_", "sgconv", "mamba")),
    ("norms (LN/RMS)   ", ("layer_norm", "rmsnorm", "RMSNorm")),
    ("act               ", ("act_and_mul", "silu", "gelu")),
    ("elementwise       ", ("elementwise", "cast", "Cast", "copy", "Copy", "cat", "Cat",
                             "fill", "Fill", "index", "triu", "where", "to_copy", "convert")),
    ("reduce/sum       ", ("reduce_kernel", "ReduceOp", "argmax", "softmax")),
    ("kv/cache misc    ", ("reshape_and_cache", "cache", "Cache")),
]
print("=" * 110)
print("DECODE-ONLY GROUPED (ms/token)")
print("=" * 110)
lm_ms = sum(e["dur"] for e in big) / 1000 / NTOK if big else 0.0
seen = {}
for name, (c, tot, durs) in stats.items():
    for gname, keys in GROUPS:
        if keys and any(k in name for k in keys):
            seen[gname] = seen.get(gname, 0.0) + tot
            break
for gname, _ in GROUPS:
    if not gname.startswith("fp16 GEMV >1ms"):
        print(f"{seen.get(gname, 0.0) / 1000 / NTOK:8.4f} ms/tok | {gname}")
print(f"{lm_ms:8.4f} ms/tok | fp16 GEMV >1ms (lm_head)")
tot_all = sum(e["dur"] for e in dec)
ung = tot_all - sum(seen.values()) - lm_ms * 1000 * NTOK
print(f"{ung / 1000 / NTOK:8.4f} ms/tok | (ungrouped)")
print(f"{tot_all / 1000 / NTOK:8.4f} ms/tok | TOTAL device kernel busy")
print(f"{len(dec) / NTOK:.0f} kernel launches/token")

# gaps within decode region
t0, t1 = dec[0]["ts"], dec[-1]["ts"] + dec[-1]["dur"]
busy = sum(e["dur"] for e in dec)
print(f"\ndecode window span {(t1 - t0) / 1e6:.3f} s, busy {busy / 1e3:.1f} ms -> "
      f"{busy / (t1 - t0) * 100:.1f}% busy; {(t1 - t0) / 1000 / NTOK:.2f} ms/token wall "
      "(PROFILER-INFLATED; live wall is 16.05 ms/token)")

streams = defaultdict(list)
for e in dec:
    streams[e["tid"]].append(e)
gaps = []
for es in streams.values():
    for a, b in zip(es, es[1:]):
        g = b["ts"] - (a["ts"] + a["dur"])
        if g > 0:
            gaps.append((g, a["name"], b["name"]))
gaps.sort(reverse=True)
print(f"total gap {sum(g for g, *_ in gaps) / 1e3:.1f} ms; gaps>300us: "
      f"{sum(1 for g, *_ in gaps if g > 300)}")
print("top 12 gaps:")
for g, a, b in gaps[:12]:
    print(f"  {g / 1e3:8.3f} ms  [{a[:44]}] -> [{b[:44]}]")

# launch-count structure per step: total launches and elementwise share
ew = [e for e in dec if any(k in e["name"] for k in
      ("elementwise", "cast", "Copy", "copy", "cat_", "fill", "index", "triu", "convert", "reduce_kernel", "rsqrt"))]
print(f"\nelementwise+norm-part launches: {len(ew) / NTOK:.0f}/token, "
      f"{sum(e['dur'] for e in ew) / 1000 / NTOK:.4f} ms/token busy")
