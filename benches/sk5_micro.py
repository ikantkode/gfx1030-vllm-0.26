"""Rung-15 final microbench: per-shape splitk cfg, old vs new, TRUE cold.

The earlier FLUSH.sum() was a no-op cold flush (256 MB of reads; the GPU
never writes the buffer, so L2 is never evicted). Here FLUSH.fill_(1.0) is a
256 MB WRITE between iters - a real L2 flush. Protocol: 25 warm iters, then
30 timed iters each preceded by fill_; median of 30. Correctness: new cfg
vs old-cfg ref, rel = max|a-b| / max|b|, screen < 0.02.

Captures the real decode GEMV weights via an awq_gemm_triton hook (M==1
only), one (x,q,s,z) per unique (N,K) shape, and times
awq_gemv_splitk_triton directly (no eager launch noise in the timed region -
events bracket the kernel+reduce only).
"""
import os

os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

import torch
from vllm import LLM, SamplingParams

print("DEVICE:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
import inspect
import vllm.model_executor.layers.quantization.awq_triton as at

assert "Rung 15" in inspect.getsource(at), "rung-15 awq_triton.py not mounted"
print("PATCHED: True", flush=True)
REAL_SK = at.awq_gemv_splitk_triton
REAL_GEMM = at.awq_gemm_triton

PROMPTS = [
    "Explain photosynthesis to a 10 year old.",
    "Write a Python function to reverse a linked list.",
    "Summarize the causes of World War I.",
    "What is 17*24? Show your reasoning.",
    "Translate 'the weather is nice today' to French.",
]

llm = LLM(
    model="/hostq/Qwen3.5-4B-AWQ-vd", quantization="awq", dtype="float16",
    max_model_len=8192, max_num_seqs=2, max_num_batched_tokens=2048,
    enable_chunked_prefill=True, gpu_memory_utilization=0.6, enforce_eager=True,
    attention_backend="ROCM_ATTN", trust_remote_code=True, generation_config="vllm",
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={"max_pixels": 1003520},
)

LOG = []


def rec(input, qweight, scales, qzeros, split_k_iters):
    if input.shape[0] == 1:
        LOG.append((input, qweight, scales, qzeros))
    return REAL_GEMM(input, qweight, scales, qzeros, split_k_iters)


at.awq_gemm_triton = rec
SP = SamplingParams(max_tokens=128, temperature=0)
llm.generate(PROMPTS, SP, use_tqdm=False)  # one full pass to capture weights
at.awq_gemm_triton = REAL_GEMM

from collections import Counter
seen = {}
shapes = Counter()
for i, q, s, z in LOG:
    key = (q.shape[1] * 8, i.shape[1])
    shapes[key] += 1
    if key not in seen:
        seen[key] = (i, q, s, z)
print("shapes (N,K): x calls:", dict(shapes), flush=True)

# per-shape counts per token (assume uniform: total / tokens generated)
ntok = sum(1 for _ in range(5 * 128))  # 5 prompts x 128
per_tok = {k: v / (5 * 128) for k, v in shapes.items()}

FLUSH = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")


def cold_time(fn, warm=25, n=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        FLUSH.fill_(1.0)          # real 256 MB write -> L2 evicted
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort()
    return ts[len(ts) // 2]


NEWCFG = {(18432, 2560): (128, 32, 4, 4),
          (12288, 2560): (128, 32, 4, 4),
          (10240, 2560): (128, 32, 4, 4),
          (2560, 9216): (128, 64, 16, 2)}
OLD = (64, 32, 16, 2)
saved = 0.0
tot_old = tot_new = 0.0
for key in sorted(seen):
    i, q, s, z = seen[key]
    N, K = key
    new = NEWCFG.get(key, OLD)
    ref = REAL_SK(i, q, s, z, *OLD, 3).float()
    out = REAL_SK(i, q, s, z, *new, 3)
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    t_old = cold_time(lambda: REAL_SK(i, q, s, z, *OLD, 3))
    t_new = cold_time(lambda: REAL_SK(i, q, s, z, *new, 3))
    c = per_tok[key]
    d_saved = max(0.0, (t_old - t_new) * c)
    saved += d_saved
    tot_old += t_old * c
    tot_new += t_new * c
    mb = (K * N * 0.5 + K * N / 64 + K * N / 256) / 1e6
    bw_new = mb / 1e3 / (t_new * 1e-3)
    print(f"N={N} K={K} x{c:.1f}/tok: old {t_old*1e3:.1f}us -> new {t_new*1e3:.1f}us "
          f"cfg={new} rel={rel:.5f} {'OK' if rel<0.02 else 'FAIL'} "
          f"newBW={bw_new:.0f}GB/s d_saved={d_saved:.4f}ms", flush=True)
print(f"\nSPLITK BLOCK: old {tot_old/1000:.3f} -> new {tot_new/1000:.3f} ms/token "
      f"(saved {saved/1000:.3f})", flush=True)
print("MICRO DONE", flush=True)
