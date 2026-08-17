"""Rung 13 offline gate: fused Gemma RMSNorm (gfx1030) vs pristine native.

Numerics: must match the native composite within the IR op's fp16 tolerance
(atol 1e-2, rtol 2e-3) on every real decode/prefill shape, including the
strided q/k split views the model actually produces.
Launch count: exactly 1 device kernel per call (vs 13 native).
Timing: fused vs native per shape + num_warps sanity for the dispatch table.
"""
import sys

sys.path.insert(0, "/hostq/gfx1030-patches")

import torch

assert torch.cuda.is_available()
dev = "cuda"

import rmsnorm_gfx1030 as r13
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

assert r13._PATCHED, "patch did not apply"
assert GemmaRMSNorm.forward_native is r13._fused_forward
print("patch ACTIVE; native captured:", r13._ORIG_FORWARD_NATIVE.__name__)

# CustomOp.__init__ requires a live vllm config. set_current_vllm_config is a
# @contextmanager whose finally restores the global on generator close, so for
# this whole-process harness just pin the module global directly.
import vllm.config.vllm as _vc
from vllm.config import VllmConfig

_vc._current_vllm_config = VllmConfig()
_vc.get_cached_compilation_config.cache_clear()


def native_ref(x, w, residual, eps):
    """Pristine GemmaRMSNorm.forward_native + ir.ops.rms_norm native body."""
    weight = w.float() + 1.0
    if residual is None:
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + eps)
        x32 = x32.to(weight.dtype) * weight
        return x32.to(x.dtype)
    x32 = x.to(torch.float32)
    x32 = x32 + residual.to(torch.float32)
    res_out = x32.to(x.dtype)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    x32 = x32.to(weight.dtype) * weight
    return x32.to(x.dtype), res_out


EPS = 1e-6
torch.manual_seed(0)

def make_cases(scale):
    """Real decode/prefill shapes from the -vd model, inputs scaled by `scale`.
    q/k stay strided views into the [rows,6144] qkv buffer exactly as the
    qkv-split produces them."""
    qkv = torch.randn(2048, 6144, device=dev, dtype=torch.float16) * (0.7 * scale)
    return [
        ("decode hidden [1,2560]", torch.randn(1, 2560, device=dev, dtype=torch.float16) * scale, False),
        ("decode hidden+res [1,2560]", torch.randn(1, 2560, device=dev, dtype=torch.float16) * scale, True),
        ("decode q view [1,16,256]", qkv[:1, :4096].view(1, 16, 256), False),
        ("decode k view [1,4,256]", qkv[:1, 4096:5120].view(1, 4, 256), False),
        ("prefill hidden [2048,2560]", torch.randn(2048, 2560, device=dev, dtype=torch.float16) * scale, False),
        ("prefill hidden+res [2048,2560]", torch.randn(2048, 2560, device=dev, dtype=torch.float16) * scale, True),
        ("prefill q view [2048,16,256]", qkv[:, :4096].view(2048, 16, 256), False),
        ("1-D edge [2560]", torch.randn(2560, device=dev, dtype=torch.float16) * scale, False),
    ]


print(f"{'case':32s} {'max_abs':>9s} {'max_rel':>9s}  verdict")
all_pass = True
for scale in (0.1, 1.0, 5.0):
    for name, x, has_res in make_cases(scale):
        N = x.shape[-1]
        w = torch.randn(N, device=dev, dtype=torch.float16) * 0.3
        res = torch.randn_like(x) * 0.5 if has_res else None
        mod = GemmaRMSNorm(N, eps=EPS).to(dev).to(torch.float16)
        with torch.no_grad():
            mod.weight.copy_(w)
            got = mod(x, res) if has_res else mod(x)
            ref = native_ref(x, w, res, EPS)
        if has_res:
            g, r = got
            rg, rr = ref
            d1 = (g.float() - rg.float()).abs()
            d2 = (r.float() - rr.float()).abs()
            d = torch.maximum(d1.max(), d2.max()).item()
            rel = (d1 / rg.float().abs().clamp_min(1e-3)).max().item()
        else:
            d = (got.float() - ref.float()).abs().max().item()
            rel = ((got.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-3)).max().item()
        ok = d <= 1e-2 and rel <= 2e-3
        all_pass &= ok
        print(f"{name+' x'+str(scale):32s} {d:9.2e} {rel:9.2e}  {'PASS' if ok else 'FAIL'}")

print("NUMERICS:", "ALL PASS" if all_pass else "FAIL")

# ---- launch count: exactly one kernel per call ----
x = torch.randn(1, 2560, device=dev, dtype=torch.float16)
mod = GemmaRMSNorm(2560, eps=EPS).to(dev).to(torch.float16)
with torch.no_grad():
    mod(x)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity

    with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(100):
            mod(x)
        torch.cuda.synchronize()
ks = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
names = {}
for e in ks:
    names[e.name] = names.get(e.name, 0) + 1
print("device kernels over 100 calls:", names)
assert sum(names.values()) == 100 and all("gemma_rmsnorm" in n for n in names), "expected exactly 1 fused kernel/call"

# ---- timing: DEVICE time via profiler (wall is CPU-dispatch dominated in
# eager; under CUDA graphs only kernel durations matter) ----
from torch.profiler import ProfilerActivity


def dev_time(fn, iters=200):
    """(device us/call, launches/call) summed over all CUDA events."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot, cnt = 0.0, 0
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            tot += e.device_time_total
            cnt += 1
    return tot / iters, cnt / iters


print(f"\n{'shape':28s} {'nat us':>7s} {'nat L':>6s} {'fus us':>7s} {'fus L':>5s} {'busy x':>7s}")
for name, x, has_res in make_cases(1.0)[:6]:
    N = x.shape[-1]
    w = torch.randn(N, device=dev, dtype=torch.float16) * 0.3
    mod = GemmaRMSNorm(N, eps=EPS).to(dev).to(torch.float16)
    with torch.no_grad():
        mod.weight.copy_(w)
        res = torch.randn_like(x) * 0.5 if has_res else None
        t_nat, l_nat = dev_time(lambda: r13._ORIG_FORWARD_NATIVE(mod, x, res))
        t_fus, l_fus = dev_time(lambda: mod(x, res))
    print(f"{name:28s} {t_nat:7.2f} {l_nat:6.1f} {t_fus:7.2f} {l_fus:5.1f} {t_nat / t_fus:6.1f}x")

# ---- num_warps sanity for the table (raw kernel, device time) ----
print()
for N in (2560, 256):
    x = torch.randn(1, N, device=dev, dtype=torch.float16)
    w = torch.randn(N, device=dev, dtype=torch.float16) * 0.3
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block = r13._next_pow2(N)
    row = []
    for wp in (1, 2, 4, 8):
        if wp * 32 > block:
            continue

        def run():
            r13._gemma_rmsnorm_kernel[(1,)](
                x, w, x, y, y, x.stride(0), x.stride(0), N, EPS,
                HAS_RES=False, BLOCK=block, num_warps=wp,
            )

        t, _ = dev_time(run, 500)
        row.append((wp, f"{t:.2f}"))
    print(f"N={N} BLOCK={block} warps sweep (device us):", row)
print("\nGATE:", "PASS" if all_pass else "FAIL")
