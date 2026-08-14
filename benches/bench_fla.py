"""Live sweep of the FLA packed-decode GDN kernel launch params on gfx1030.

Model shapes: H=16, HV=32, K=128, V=128, B=1 (single-token decode).
Stock config: BV=32, num_warps=1, num_stages=3.
"""

import time

import torch
import triton

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode_kernel as kern,
)

B, H, HV, K, V = 1, 16, 32, 128, 128
scale = 1.0 / (K**0.5)

torch.manual_seed(0)
qkv = torch.randn(B, H * K * 2 + HV * V, device="cuda", dtype=torch.float16)
a = torch.randn(B, HV, device="cuda", dtype=torch.float32)
b = torch.randn(B, HV, device="cuda", dtype=torch.float32)
A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)
state0 = torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.1
out = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.float16)
idx = torch.zeros(B, device="cuda", dtype=torch.int32)


def launch(BV, W, S, o, st):
    BK = 128
    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)
    kern[grid](
        mixed_qkv=qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, o=o,
        h0=st, ht=st, ssm_state_indices=idx, scale=scale,
        stride_mixed_qkv_tok=qkv.stride(0), stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0), stride_init_state_token=st.stride(0),
        stride_final_state_token=st.stride(0), stride_indices_seq=idx.stride(0),
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=False,
        num_warps=W, num_stages=S,
    )


# reference: stock config
ref_o = torch.empty_like(out)
ref_s = state0.clone()
launch(32, 1, 3, ref_o, ref_s)
torch.cuda.synchronize()

CONFIGS = [
    (32, 1, 3),  # stock
    (32, 2, 3),
    (32, 4, 3),
    (32, 8, 3),
    (64, 2, 3),
    (64, 4, 3),
    (64, 8, 3),
    (32, 4, 2),
    (32, 4, 4),
]

for (BV, W, S) in CONFIGS:
    try:
        o = torch.empty_like(out)
        st = state0.clone()
        launch(BV, W, S, o, st)
        torch.cuda.synchronize()
        odiff = (o.float() - ref_o.float()).abs().max().item()
        sdiff = (st - ref_s).abs().max().item()
        for _ in range(20):
            launch(BV, W, S, o, st)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(500):
            launch(BV, W, S, o, st)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t) / 500 * 1e6
        print(f"BV={BV:3d} W={W} S={S}: {us:7.1f} us/call   out_diff={odiff:.2e} state_diff={sdiff:.2e}", flush=True)
    except Exception as e:
        print(f"BV={BV} W={W} S={S}: FAIL {type(e).__name__} {str(e)[:90]}", flush=True)
