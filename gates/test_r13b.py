"""Rung 13 offline model-level sanity (§21): patch active under the real
serving arg mirror (graphs ON), generation coherent, no dispatch surprises.

Mirrors prof_r12.py's LLM args exactly (the SHIPPED path), minus profiling.
"""
import os
import sys
import time

os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
sys.path.insert(0, "/hostq/gfx1030-patches")

import torch

print("DEVICE:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)

import rmsnorm_gfx1030 as r13

assert r13._PATCHED, "patch did not apply"
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

assert GemmaRMSNorm.forward_cuda is r13._fused_forward
print(">>> patch ACTIVE in this process", flush=True)

from vllm import LLM, SamplingParams

llm = LLM(
    model="/hostq/Qwen3.5-4B-AWQ-vd", quantization="awq", dtype="float16",
    max_model_len=8192, max_num_seqs=2, max_num_batched_tokens=2048,
    enable_chunked_prefill=True, gpu_memory_utilization=0.9,
    attention_backend="ROCM_ATTN", trust_remote_code=True, generation_config="vllm",
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={"max_pixels": 1003520},
)
llm.generate(["hello"], SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)

prompts = [
    "Write a long detailed essay about the history of computing.",
    "The capital of France is",
    "Explain photosynthesis to a ten-year-old in three sentences.",
]
for p in prompts:
    out = llm.generate([p], SamplingParams(max_tokens=64, temperature=0, ignore_eos=True), use_tqdm=False)
    print(f"\n=== {p!r}\n{out[0].outputs[0].text}", flush=True)

t0 = time.perf_counter()
out = llm.generate(
    ["Write a long detailed essay about the history of computing."],
    SamplingParams(max_tokens=256, temperature=0, ignore_eos=True), use_tqdm=False,
)
dt = time.perf_counter() - t0
n = len(out[0].outputs[0].token_ids)
print(f"\n>>> offline decode: {n} tokens in {dt:.2f}s = {n / dt:.1f} TPS (graphs on, GPU 1)")
print(">>> SANITY: generation above must be coherent English")
