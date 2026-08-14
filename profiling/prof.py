import os
from vllm import LLM, SamplingParams

print(">>> loading model (single-process for profiling)", flush=True)
llm = LLM(
    model="/model",
    quantization="awq",
    dtype="float16",
    enforce_eager=True,
    max_model_len=2048,
    gpu_memory_utilization=0.9,
    attention_backend="ROCM_ATTN",
)
print(">>> warmup (compile kernels)", flush=True)
llm.generate(["hello"], SamplingParams(max_tokens=4), use_tqdm=False)
print(">>> decode-heavy run (256 tok) - decode kernels dominate the stats", flush=True)
llm.generate(
    ["Write a long detailed essay about the history of computing."],
    SamplingParams(max_tokens=256, ignore_eos=True),
    use_tqdm=False,
)
print(">>> DONE", flush=True)
