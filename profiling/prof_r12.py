"""Rung-12 profiler capture: the SHIPPED execution path, steady-state decode.

Live server (GPU 0) exposes no /start_profile in this build, so per §21 we
capture offline on GPU 1 with every prod serve arg mirrored — including
VLLM_USE_BREAKABLE_CUDAGRAPH=1 (profile what ships, NOT enforce_eager).
Supervisor schedule(wait=2, warmup=2, active=5, repeat=1) intends "capture
steady state, skip cold start"; offline LLM() has no per-engine-step hook to
drive profiler.step(), so the equivalent is: unprofiled warmup generates,
then full-window capture of the 256-token temp-0 decode.

Invocation (each env var cost a failed attempt to discover):
  docker exec -e HIP_VISIBLE_DEVICES=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
       -e VLLM_ENABLE_V1_MULTIPROCESSING=0 quant-run python3 /qwork/prof_r12.py
- HIP_VISIBLE_DEVICES (NOT CUDA_VISIBLE_DEVICES): CVD on ROCm spins forever
  in the multimodal encoder-profiling path (~17 min at 103% CPU, idle GPU).
- VLLM_ENABLE_V1_MULTIPROCESSING=0: V1 EngineCore otherwise runs the forward
  in a child process and the parent's profiler captures nothing (29 us total).
- torch renamed FunctionEventAvg.cuda_time_total -> device_time_total; the
  CSV writer uses getattr fallback so it works on both.
- kineto/ROCTracer adds ~60 us/launch CPU tax: wall in the trace is ~6.5x
  live (103.5 vs 16.05 ms/token). Kernel DURATIONS are honest; gaps are not
  (use analyze_r12b.py totals + live-wall subtraction instead).
"""
import os
os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

import torch


def main():
    print("DEVICE:", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/hostq/Qwen3.5-4B-AWQ-vd", quantization="awq", dtype="float16",
        max_model_len=8192, max_num_seqs=2, max_num_batched_tokens=2048,
        enable_chunked_prefill=True, gpu_memory_utilization=0.9,
        attention_backend="ROCM_ATTN", trust_remote_code=True, generation_config="vllm",
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": 1003520},
    )
    # warmup: graphs are captured at init; this warms sampler + python paths
    llm.generate(["hello"], SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)
    llm.generate(["Write a long detailed essay about the history of computing."],
                 SamplingParams(max_tokens=64, temperature=0, ignore_eos=True), use_tqdm=False)
    print(">>> WARMUP DONE, starting profiled 256-token decode", flush=True)

    from torch.profiler import ProfilerActivity

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as prof:
        out = llm.generate(
            ["Write a long detailed essay about the history of computing."],
            SamplingParams(max_tokens=256, temperature=0, ignore_eos=True),
            use_tqdm=False,
        )
    print("SAMPLE:", repr(out[0].outputs[0].text[:120]), flush=True)

    ka = prof.key_averages()
    print(ka.table(sort_by="cuda_time_total", row_limit=60), flush=True)

    import csv
    with open("/qwork/prof_r12_kernels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "count", "cuda_total_us", "cpu_total_us"])
        for e in ka:
            if e.count:
                # torch renamed cuda_time_total -> device_time_total
                gpu_us = getattr(e, "device_time_total", None)
                if gpu_us is None:
                    gpu_us = getattr(e, "cuda_time_total", 0)
                w.writerow([e.key, e.count, gpu_us, e.cpu_time_total])

    prof.export_chrome_trace("/qwork/prof_r12_trace.json")
    print("TRACE_EXPORTED /qwork/prof_r12_trace.json", flush=True)


if __name__ == "__main__":
    main()
