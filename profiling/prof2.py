"""Decode kernel attribution via torch.profiler (works on ROCm, unlike rocprof v1)."""

from vllm import LLM, SamplingParams

print(">>> loading (enforce_eager so individual kernels are visible)", flush=True)
llm = LLM(
    model="/model", quantization="awq", dtype="float16", enforce_eager=True,
    max_model_len=2048, gpu_memory_utilization=0.9, attention_backend="ROCM_ATTN",
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={"max_pixels": 1003520},
)
llm.generate(["hello"], SamplingParams(max_tokens=4), use_tqdm=False)
print(">>> profiling 96-token decode", flush=True)

import torch
from torch.profiler import ProfilerActivity

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    llm.generate(
        ["Write a long detailed essay about the history of computing."],
        SamplingParams(max_tokens=96, ignore_eos=True),
        use_tqdm=False,
    )

print(">>> top kernels by CUDA time (decode window)", flush=True)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
print(">>> DONE", flush=True)
