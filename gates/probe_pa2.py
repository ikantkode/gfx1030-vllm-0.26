"""Rung-14 candidate probe (investigate-only): why is kernel_paged_attention_2d
183 us/call live (8 calls/token = 1.46 ms), and is it a *config* problem?

Live dispatch facts established:
  - custom paged_attention_rocm path: gated OFF for gfx1030 (arch gate), AND
    unsupported head_size=256, AND non-pow2 block 528 -> Triton forced.
  - Triton path: TRITON_BLOCK_SIZE forced to 32 (non-pow2 physical block),
    no num_warps/num_stages at launch (defaults 4/3?), grid (1, 4).

Plan: run the offline LLM (serving arg mirror, GPU 1), wrap
chunked_prefill_paged_decode, snapshot the LAST deep-decode call's tensors,
then replay: (a) wrapper as-is = baseline, (b) raw kernel with swept
BLOCK_SIZE x num_warps x num_stages. Verify vs baseline + fp32 reference.
Timing = torch.profiler device time (test_r13 standard).
"""
import os
import sys

# eager probe: the wrapper must see every decode call; kernel device-time
# (what we measure) is identical under graphs — only launch overhead differs.
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

import torch

print("DEVICE:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)

import vllm.v1.attention.ops.chunked_prefill_paged_decode as cppd_mod
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as ORIG_CPPD,
)

CAP = {}
POINTS = []  # captures at increasing seq depth
_ORIG_FN = ORIG_CPPD
_calls = [0]
_last_cap_seq = [0]


def _capture(**kw):
    # keep only the LAST deep single-token call (seq well past 200)
    CAP.clear()
    CAP.update(kw)
    CAP["_meta"] = dict(
        key_stride=tuple(kw["key_cache"].stride()),
        val_stride=tuple(kw["value_cache"].stride()),
        key_shape=tuple(kw["key_cache"].shape),
        val_shape=tuple(kw["value_cache"].shape),
        q_shape=tuple(kw["query"].shape),
        block_dtype=str(kw["block_table"].dtype),
    )
    CAP["query"] = kw["query"].clone()
    CAP["output_ref"] = None  # filled post-call
    CAP["block_table"] = kw["block_table"].clone()
    CAP["seq_lens"] = kw["seq_lens"].clone()
    CAP["query_start_loc"] = kw["query_start_loc"].clone()


def _wrapped(*args, **kwargs):
    r = _ORIG_FN(*args, **kwargs)
    if torch.cuda.is_current_stream_capturing():
        return r
    if kwargs.get("max_query_len", 1) == 1:
        sl = int(kwargs["seq_lens"][0])
        if sl > 200 and sl - _last_cap_seq[0] >= 40 and len(POINTS) < 6:
            _capture(**kwargs)
            CAP["output_ref"] = kwargs["output"].clone()
            POINTS.append({k: CAP[k] for k in
                           ("query", "output_ref", "block_table", "seq_lens",
                            "query_start_loc", "key_cache", "value_cache",
                            "kv_cache_dtype", "max_seq_len", "max_query_len",
                            "k_scale", "v_scale", "alibi_slopes",
                            "sliding_window", "sm_scale", "output_scale",
                            "sinks")})
            _last_cap_seq[0] = sl
    return r


cppd_mod.chunked_prefill_paged_decode = _wrapped
import vllm.v1.attention.backends.rocm_attn as ra_mod

if hasattr(ra_mod, "chunked_prefill_paged_decode"):
    ra_mod.chunked_prefill_paged_decode = _wrapped
print(">>> wrapper installed", flush=True)

from vllm import LLM, SamplingParams

llm = LLM(
    model="/hostq/Qwen3.5-4B-AWQ-vd", quantization="awq", dtype="float16",
    max_model_len=8192, max_num_seqs=2, max_num_batched_tokens=2048,
    enable_chunked_prefill=True, gpu_memory_utilization=0.6, enforce_eager=True,
    attention_backend="ROCM_ATTN", trust_remote_code=True, generation_config="vllm",
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={"max_pixels": 1003520},
)
llm.generate(["hello"], SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)
out = llm.generate(
    ["Write a long detailed essay about the history of computing."],
    SamplingParams(max_tokens=300, temperature=0, ignore_eos=True), use_tqdm=False,
)
print("gen tokens:", len(out[0].outputs[0].token_ids), flush=True)
assert POINTS, "no deep-decode call captured"
m = CAP["_meta"]  # grab before clear (POINTS entries don't carry it)
CAP.clear()
CAP.update(POINTS[0])
print(f">>> CAPTURED {len(POINTS)} points, seq {[int(P['seq_lens'][0]) for P in POINTS]}", flush=True)
print(f">>> meta q{m['q_shape']} k{m['key_shape']} s{m['key_stride']} v{m['val_shape']} s{m['val_stride']} bt={CAP['block_table'].shape} {m['block_dtype']}", flush=True)

# ---- replay machinery ----
from vllm.v1.attention.ops.chunked_prefill_paged_decode import kernel_paged_attention_2d
import triton
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity

def dev_time(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = sum(e.device_time_total for e in p.key_averages()
              if e.device_type == DeviceType.CUDA and e.device_time_total > 0)
    return tot / iters

def replay_wrapper(P):
    query = P["query"]
    out = torch.empty_like(query)
    ORIG_CPPD(
        query=query, key=None, value=None, output=out,
        kv_cache_dtype=P["kv_cache_dtype"],
        key_cache=P["key_cache"], value_cache=P["value_cache"],
        block_table=P["block_table"],
        query_start_loc=P["query_start_loc"], seq_lens=P["seq_lens"],
        max_seq_len=P["max_seq_len"], max_query_len=P["max_query_len"],
        k_scale=P["k_scale"], v_scale=P["v_scale"],
        alibi_slopes=P["alibi_slopes"], sliding_window=P["sliding_window"],
        sm_scale=P["sm_scale"], output_scale=P["output_scale"],
        sinks=P["sinks"], is_block_table_ptr=False,
    )
    return out

def raw_call(P, block=None, warps=None, stages=None):
    query = P["query"]
    key_cache, value_cache = P["key_cache"], P["value_cache"]
    block_table, qsl = P["block_table"], P["query_start_loc"]
    seq_lens = P["seq_lens"]
    num_tokens, num_query_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    nqpk = num_query_heads // num_kv_heads
    nqpk_pad = max(triton.next_power_of_2(nqpk), 16)
    bs = value_cache.shape[3]
    is_pow2 = bs & (bs - 1) == 0
    TB = min(bs, 128) if is_pow2 else 32
    if block is not None:
        TB = block
    out = torch.empty_like(query)
    pbt = block_table.to(torch.int32)
    kw = {}
    if warps:
        kw["num_warps"] = warps
    if stages:
        kw["num_stages"] = stages
    kernel_paged_attention_2d[(1, num_kv_heads)](
        output_ptr=out, query_ptr=query,
        key_cache_ptr=key_cache, value_cache_ptr=value_cache,
        sink_ptr=P["sinks"], block_tables_ptr=pbt, seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=P["alibi_slopes"], scale=P["sm_scale"],
        k_scale=P["k_scale"], v_scale=P["v_scale"], out_scale_inv=1.0,
        num_query_heads=num_query_heads, num_queries_per_kv=nqpk,
        num_queries_per_kv_padded=nqpk_pad,
        block_table_stride=pbt.stride(0),
        query_stride_0=query.stride(0), query_stride_1=query.stride(1),
        output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        BLOCK_SIZE=TB, PHYSICAL_BLOCK_SIZE=bs,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=False, SLIDING_WINDOW=0,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0), stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2), stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0), stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2), stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=True, query_start_len_ptr=qsl,
        USE_SINKS=False, USE_FP8=False, **kw,
    )
    return out

# ---- per-seq-depth: live config (BLOCK=32, warps default) vs warps=8 ----
# live_kernel = kernel-only at live config (matches how the 183us live figure
# was measured); live_wrap = full wrapper path (includes any prep kernels).
print(f"\n{'seq_len':>7s} {'live_wrap':>9s} {'live_kern':>9s} {'w8':>8s} {'save/8calls':>11s}  fidelity")
tot_save = 0.0
for P in POINTS:
    sl = int(P["seq_lens"][0])
    us_wrap = dev_time(lambda: replay_wrapper(P))
    us_live = dev_time(lambda: raw_call(P, None, None))
    us_w8 = dev_time(lambda: raw_call(P, 32, 8))
    d = (raw_call(P, 32, 8).float() - P["output_ref"].float()).abs().max().item()
    tot_save += us_live - us_w8
    print(f"{sl:7d} {us_wrap:9.2f} {us_live:9.2f} {us_w8:8.2f} {8*(us_live-us_w8)/1000:11.3f}  max_abs={d:.2e}", flush=True)
avg = tot_save / len(POINTS)
print(f"\n>>> mean kernel-only saving per token over captured depths: 8 * {avg:.1f} us = {8*avg/1000:.3f} ms", flush=True)

# ---- trimmed config check on the deepest point ----
P = POINTS[-1]
print(f"\nconfig check at seq={int(P['seq_lens'][0])}:")
for tag, b, w in (("live (BLOCK=32,w=def)", None, None), ("BLOCK=32,w8", 32, 8),
                  ("BLOCK=64,w8", 64, 8), ("BLOCK=32,w16", 32, 16)):
    try:
        us = dev_time(lambda b=b, w=w: raw_call(P, b, w))
        d = (raw_call(P, b, w).float() - P["output_ref"].float()).abs().max().item()
        print(f"  {tag:24s} {us:8.2f} us  max_abs={d:.2e}", flush=True)
    except Exception as e:
        print(f"  {tag:24s} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)
