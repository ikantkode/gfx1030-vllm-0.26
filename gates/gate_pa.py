"""Rung-14 offline gate: 5-prompt A/B (PLAN 2.5 set) through the offline LLM
on GPU 1. Run once WITHOUT the patched chunked_prefill_paged_decode mount
(baseline) and once WITH it; compare token IDs + text. temperature=0.
"""
import os

os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

import hashlib

import torch
from vllm import LLM, SamplingParams

print("DEVICE:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)

# confirm which variant is live (patched file has the rung-14 marker)
import inspect
import vllm.v1.attention.ops.chunked_prefill_paged_decode as cppd

src = inspect.getsource(cppd)
print("PATCHED:", "rung 14" in src, flush=True)

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
outs = llm.generate(PROMPTS, SamplingParams(max_tokens=128, temperature=0), use_tqdm=False)
for i, o in enumerate(outs):
    ids = o.outputs[0].token_ids
    h = hashlib.sha1(torch.tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest()[:12]
    print(f"PROMPT {i+1} ids[:128] sha1={h} n={len(ids)}", flush=True)
    print(f"TEXT {i+1} <<<{o.outputs[0].text}>>>", flush=True)
