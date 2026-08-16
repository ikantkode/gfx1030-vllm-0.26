"""Extract ONE decode step's kernel sequence (run-length encoded) from the
r12 trace: the repeating launch topology fusion kernels would target."""

import json

TRACE = "/qwork/prof_r12_trace.json"

with open(TRACE) as f:
    tr = json.load(f)
ev = [e for e in tr["traceEvents"]
      if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
ev.sort(key=lambda e: e["ts"])

# step boundaries: every 128th splitk kernel (tracer stalls make gap
# detection unusable; count-based segmentation is exact)
sk_idx = [i for i, e in enumerate(ev) if "awq_gemv_splitk_kernel" in e["name"]]
bounds = sk_idx[::128]
print(f"splitk kernels {len(sk_idx)} -> {len(bounds)} step starts")
# take a middle step
k = len(bounds) // 2
s, e_end = bounds[k], bounds[k + 1]
step = ev[s:e_end]
print(f"step [{k}]: {len(step)} kernels, span {(step[-1]['ts'] + step[-1]['dur'] - step[0]['ts']) / 1000:.2f} ms "
      f"(profiler-inflated), busy {sum(x['dur'] for x in step) / 1000:.3f} ms\n")

# run-length encode by short name
def short(n):
    for key, tag in [
        ("awq_gemv_splitk_kernel", "INT4_SPLITK_GEMV"),
        ("awq_gemv_splitk_reduce", "INT4_SPLITK_REDUCE"),
        ("LLGemm1_kernel", "FP16_GEMV_LLMM1"),
        ("paged_attention_2d", "PAGED_ATTN"),
        ("fused_recurrent_gated_delta_rule", "FLA_RECURRENT"),
        ("causal_conv1d_update", "CAUSAL_CONV1D"),
        ("act_and_mul", "ACT_AND_MUL"),
        ("layer_norm_fwd", "FLA_LAYERNORM"),
        ("reshape_and_cache", "KV_CACHE_WRITE"),
        ("mrope", "MROPE"),
        ("rocclr_copyBuffer", "COPYBUF"),
        ("vectorized_elementwise_kernel<4, at::native::BinaryFunctor<c10::Half, c10::Half, c10::Half, at::native::mul_kernel<", "EW mul(half)"),
        ("BinaryFunctor<c10::Half, c10::Half, c10::Half, at::native::add_kernel", "EW add(half)"),
        ("BinaryFunctor", "EW binary"),
        ("UnaryFunctor", "EW unary"),
        ("reduce_kernel", "REDUCE(RMSNorm/argmax)"),
        ("rsqrt", "EW rsqrt"),
        ("elementwise_kernel_manual_unroll", "EW manual"),
        ("vectorized_elementwise_kernel", "EW vectorized"),
        ("index_elementwise", "EW index"),
        ("argmax", "ARGMAX"),
        ("CUDAFunc", "EW cudafunc"),
        ("FillFunctor", "EW fill"),
        ("direct_copy", "COPY direct"),
        ("copy_", "COPY"),
        ("triu", "EW triu"),
    ]:
        if key in n:
            return tag
    return n[:44]

runs = []
cur_name, cur_start, n, tot = short(step[0]["name"]), 0, 0, 0.0
for i, x in enumerate(step):
    nm = short(x["name"])
    if nm != cur_name:
        runs.append((cur_name, n, tot))
        cur_name, cur_start, n, tot = nm, i, 0, 0.0
    n += 1
    tot += x["dur"]
runs.append((cur_name, n, tot))
print(f"{'run':38s} {'cnt':>4s} {'busy us':>9s}")
for nm, n, tot in runs:
    print(f"{nm:38s} {n:4d} {tot:9.1f}")
print(f"\ntotal runs {len(runs)}")
