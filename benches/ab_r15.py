"""Rung-15 in-process A/B + offline gate on the LIVE build (GPU 0).

The mounted awq_triton.py is the rung-15 per-shape splitk table. The
baseline (rung-14) is reproduced in the same process by proxying
awq_gemv_splitk_triton to force the old single config (64, 32, 16, 2).
Greedy + temperature 0 + no spec-decode => deterministic, so the two runs
are comparable. Output:
  - baseline token shas (new anchors) + text -> new_outputs_r15_base.txt
  - patched text -> new_outputs_r15_new.txt
  - first divergence + context per prompt
  - per-shape cold med-30 new vs old cfg, rel<0.02 screen
  - isolated device-sum of the captured per-token GEMV sequence (new dispatch)
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
SP = SamplingParams(max_tokens=128, temperature=0)

import hashlib


def sha(ids):
    return hashlib.sha1(torch.tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest()[:12]


def gen():
    outs = llm.generate(PROMPTS, SP, use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in outs]


print("\n=== BASELINE (proxy: old single cfg 64/32/16/W2) ===", flush=True)
at.awq_gemv_splitk_triton = lambda inp, q, s, z, bn, bk, sp, w, ns: REAL_SK(
    inp, q, s, z, 64, 32, 16, 2, 3
)
base_ids = gen()
tok = llm.get_tokenizer()
base_texts = [tok.decode(ids) for ids in base_ids]
with open("/tmp/new_outputs_r15_base.txt", "w") as f:
    f.write("\n".join(base_texts))
for i, ids in enumerate(base_ids, 1):
    print(f"P{i} sha1={sha(ids)}", flush=True)

print("\n=== PATCHED (per-shape table) ===", flush=True)
at.awq_gemv_splitk_triton = REAL_SK
new_ids = gen()
new_texts = [tok.decode(ids) for ids in new_ids]
with open("/tmp/new_outputs_r15_new.txt", "w") as f:
    f.write("\n".join(new_texts))

for i, (b, n) in enumerate(zip(base_ids, new_ids), 1):
    if b == n:
        print(f"\nP{i}: IDENTICAL 128/128 tokens", flush=True)
        continue
    d = next(j for j in range(128) if b[j] != n[j])
    print(f"\nP{i}: diverge at token {d}/128")
    print(f"  base[{d-6}:{d+8}] : {tok.decode(b[max(0,d-6):d+8])!r}")
    print(f"  new [{d-6}:{d+8}] : {tok.decode(n[max(0,d-6):d+8])!r}")
    print(f"  base tail: ...{tok.decode(b[d:d+60])!r}")
    print(f"  new  tail: ...{tok.decode(n[d:d+60])!r}")

# ---------------------------------------------------------------- measurement
print("\n=== MEASUREMENT (new dispatch, isolated replay) ===", flush=True)

LOG = []
REAL_GEMM = at.awq_gemm_triton


def rec(input, qweight, scales, qzeros, split_k_iters):
    if input.shape[0] == 1:
        LOG.append((input, qweight, scales, qzeros))
    return REAL_GEMM(input, qweight, scales, qzeros, split_k_iters)


at.awq_gemm_triton = rec
gen()  # warmup
LOG.clear()
out_ids = gen()
per_tok = len(LOG) // sum(len(ids) for ids in out_ids)
SEQ = LOG[-per_tok:]  # last token's calls (steady state)
print(f"captured {len(LOG)} calls -> {per_tok}/token (steady-state token)", flush=True)
at.awq_gemm_triton = REAL_GEMM

from collections import Counter
shapes = Counter((q.shape[1] * 8, i.shape[1]) for i, q, s, z in SEQ)
print("per-token shapes:", dict(shapes), flush=True)
seen = {}
for i, q, s, z in SEQ:
    key = (q.shape[1] * 8, i.shape[1])
    if key not in seen:
        seen[key] = (i, q, s, z)
print("unique shapes:", list(seen), flush=True)

import torch.profiler as tp
torch.cuda.synchronize()
for _ in range(3):
    for i, q, s, z in SEQ:
        at.awq_gemm_triton(i, q, s, z, 0)
torch.cuda.synchronize()
with tp.profile(activities=[tp.ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        for i, q, s, z in SEQ:
            at.awq_gemm_triton(i, q, s, z, 0)
    torch.cuda.synchronize()
dev = 0.0
for e in prof.key_averages():
    if e.device_type == tp.DeviceType.CUDA and e.device_time_total > 0:
        dev += e.device_time_total
print(f"new dispatch device-sum: {dev/5000:.3f} ms/token "
      f"(5 reps x {len(SEQ)} calls)", flush=True)

FLUSH = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")
FLUSH.fill_(1.0)


def cold_time(fn, n=30):
    ts = []
    for _ in range(n):
        FLUSH.sum()
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
saved = 0.0
for key in sorted(seen):
    i, q, s, z = seen[key]
    N, K = key
    new = NEWCFG.get(key, (64, 32, 16, 2))
    ref = REAL_SK(i, q, s, z, 64, 32, 16, 2, 3).float()
    out = REAL_SK(i, q, s, z, *new, 3)
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    t_old = cold_time(lambda: REAL_SK(i, q, s, z, 64, 32, 16, 2, 3))
    t_new = cold_time(lambda: REAL_SK(i, q, s, z, *new, 3))
    cnt = shapes[key]
    d_saved = max(0.0, (t_old - t_new) * cnt / 1000)
    saved += d_saved
    print(f"shape N={N} K={K} x{cnt}: old {t_old*1e3:.1f}us -> "
          f"new {t_new*1e3:.1f}us cfg={new} rel={rel:.5f} "
          f"{'OK' if rel < 0.02 else 'FAIL'} d_saved={d_saved:.4f}ms",
          flush=True)
print(f"GATE DONE; per-token cold saving: {saved:.3f} ms", flush=True)
