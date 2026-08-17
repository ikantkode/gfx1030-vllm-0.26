# SPDX-License-Identifier: Apache-2.0
# Rung 13 — fused Gemma-style RMSNorm for gfx1030 (RDNA2).
#
# WHY: Qwen3.5 aliases GemmaRMSNorm (gain 1+w). Its forward runs
# `self.weight.float() + 1.0` per call (2 kernels) then the IR op
# `rms_norm`, whose native impl is a 13-launch ATen chain (cast, pow,
# mean-reduce, rsqrt, muls, casts) x81 norms/token = ~1.95 ms busy +
# ~1050 launches/token at 62 TPS (Entry 26). The compiled `_C` fused
# kernels exist in this build but their IR provider gate requires
# weight.dtype == x.dtype, and Gemma passes an fp32 weight against
# fp16 activations -> always rejected -> native chain, and no IR
# priority is ever configured in this deployment anyway.
#
# WHAT: one Triton kernel per (N, residual?) shape: load row fp16->fp32,
# optional residual add, sum(x*x)/N, rsqrt, multiply by (1+w) computed
# in fp32 from the RAW fp16 weight (never precompute an fp16 (1+w) —
# see the tier-3 LN-fold lesson), single fp16 store. Matches the native
# op order (x32*inv)*w -> .to(fp16); only the fp32 reduction tree order
# differs (IR tolerance for fp16: atol 1e-2 / rtol 2e-3).
#
# INSTALL: file-mount into vllm/model_executor/layers/ and import from
# the mounted awq_triton.py. Self-applies the GemmaRMSNorm patch at
# import; unsupported shapes fall back to the original native path.

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_MAX_N = 8192  # single-block masked kernel; all real N (2560, 256) fit


@triton.jit
def _gemma_rmsnorm_kernel(
    X, W, R, Y, RES,
    stride_x, stride_r,
    N, eps,
    HAS_RES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
    var = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    y = (x * inv) * w
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)
    if HAS_RES:
        tl.store(RES + row * N + cols, x.to(RES.dtype.element_ty), mask=mask)


def _warps_for(block: int) -> int:
    # device-time swept on gfx1030: N=2560/BLOCK 4096 -> 8 warps (2.06 us),
    # N=256/BLOCK 256 -> 1 warp (1.83 us)
    if block >= 2048:
        return 8
    if block >= 512:
        return 4
    return 1


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def gemma_rmsnorm_fused(x: torch.Tensor, weight: torch.Tensor, residual, eps: float):
    """Fused replacement for GemmaRMSNorm.forward. Callers must have checked
    the gate in _fused_forward (cuda, dtype, N<=8192, stride(-1)==1)."""
    N = x.shape[-1]
    x2 = x.reshape(-1, N)  # view for decode shapes; harmless copy if strided
    rows = x2.shape[0]
    block = _next_pow2(N)
    warps = _warps_for(block)
    # outputs must be dense (empty_like would inherit a strided view's layout)
    y = torch.empty(x2.shape, dtype=x.dtype, device=x.device)
    if residual is None:
        _gemma_rmsnorm_kernel[(rows,)](
            x2, weight, x2, y, y, x2.stride(0), x2.stride(0),
            N, eps, HAS_RES=False, BLOCK=block, num_warps=warps,
        )
        return y.view(x.shape)
    r2 = residual.reshape(-1, N)
    res = torch.empty(r2.shape, dtype=residual.dtype, device=residual.device)
    _gemma_rmsnorm_kernel[(rows,)](
        x2, weight, r2, y, res, x2.stride(0), r2.stride(0),
        N, eps, HAS_RES=True, BLOCK=block, num_warps=warps,
    )
    return y.view(x.shape), res.view(residual.shape)


def _fused_forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    w = self.weight
    if (
        x.is_cuda
        and x.dim() >= 1
        and x.shape[-1] <= _MAX_N
        and x.stride(-1) == 1
        and x.dtype in _SUPPORTED_DTYPES
        and w.dtype == x.dtype
        and w.dim() == 1
        and w.shape[0] == x.shape[-1]
    ):
        return gemma_rmsnorm_fused(x, w, residual, self.variance_epsilon)
    return _ORIG_FORWARD_NATIVE(self, x, residual)


def apply_patch() -> bool:
    """Monkey-patch GemmaRMSNorm (aliased as Qwen3_5RMSNorm / Qwen3NextRMSNorm)
    to route to the fused kernel. Idempotent; returns True if active."""
    global _ORIG_FORWARD_NATIVE, _PATCHED
    if _PATCHED:
        return True
    try:
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        _ORIG_FORWARD_NATIVE = GemmaRMSNorm.forward_native
        # forward_cuda/forward_hip delegate to forward_native in stock vLLM;
        # CustomOp binds _forward_method at __init__, and this patch runs at
        # import time (before model construction), so rebinding both covers
        # every dispatch route.
        GemmaRMSNorm.forward_native = _fused_forward
        GemmaRMSNorm.forward_cuda = _fused_forward
        _PATCHED = True
        return True
    except Exception as e:  # pragma: no cover - loud but non-fatal
        print(f"[rmsnorm_gfx1030] patch NOT applied, using native: {e!r}")
        return False


_ORIG_FORWARD_NATIVE = None
_PATCHED = False
_PATCHED = apply_patch()
