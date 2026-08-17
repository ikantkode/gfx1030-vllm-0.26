"""Rung-14 offline gate, take 2: baseline vs patched IN ONE PROCESS.

The patched file is mounted (num_warps=8 at launch). To reproduce baseline
behavior without a second container, proxy kernel_paged_attention_2d and pop
num_warps from the launch kwargs -> JIT default (4) = stock behavior. The
proxy run's token-ID sha1 MUST match gate_pa_baseline.txt (eager, greedy,
deterministic) — that self-validates the proxy. Then re-enable the real
kernel (warps=8), regenerate, and report first divergence + both texts.
"""
import os

os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

import torch
from vllm import LLM, SamplingParams

print("DEVICE:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)

import inspect
import vllm.v1.attention.ops.chunked_prefill_paged_decode as cppd

assert "rung 14" in inspect.getsource(cppd), "patched file not mounted"
print("PATCHED: True", flush=True)
REAL_KERNEL = cppd.kernel_paged_attention_2d


class StripWarps:
    def __getitem__(self, grid):
        def launch(*a, **kw):
            kw.pop("num_warps", None)  # -> JIT default = stock behavior
            return REAL_KERNEL[grid](*a, **kw)

        return launch


PROMPTS = [
    "Explain photosynthesis to a 10 year old.",
    "Write a Python function to reverse a linked list.",
    "Summarize the causes of World War I.",
    "What is 17*24? Show your reasoning.",
    "Translate 'the weather is nice today' to French.",
]
EXPECTED_BASE_SHA = {
    1: "e4d6d120d503", 2: "ef68152b2b95", 3: "53b5586b4ea3",
    4: "23078ba2ca3c", 5: "52a1350a302f",
}

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


print("\n=== BASELINE (proxy, num_warps stripped) ===", flush=True)
cppd.kernel_paged_attention_2d = StripWarps()
base_ids = gen()
for i, ids in enumerate(base_ids, 1):
    s = sha(ids)
    ok = "MATCH" if s == EXPECTED_BASE_SHA[i] else f"MISMATCH(expect {EXPECTED_BASE_SHA[i]})"
    print(f"P{i} sha1={s} {ok}", flush=True)

print("\n=== PATCHED (warps=8) ===", flush=True)
cppd.kernel_paged_attention_2d = REAL_KERNEL
new_ids = gen()

tok = llm.get_tokenizer()
for i, (b, n) in enumerate(zip(base_ids, new_ids), 1):
    if b == n:
        print(f"\nP{i}: IDENTICAL 128/128 tokens", flush=True)
        continue
    d = next(j for j in range(128) if b[j] != n[j])
    print(f"\nP{i}: diverge at token {d}/128", flush=True)
    print(f"  base[{d-6}:{d+8}] : {tok.decode(b[max(0,d-6):d+8])!r}", flush=True)
    print(f"  new [{d-6}:{d+8}] : {tok.decode(n[max(0,d-6):d+8])!r}", flush=True)
    print(f"  base tail: ...{tok.decode(b[d:d+60])!r}", flush=True)
    print(f"  new  tail: ...{tok.decode(n[d:d+60])!r}", flush=True)
