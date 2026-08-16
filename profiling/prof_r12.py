"""Rung-12 profiler capture: the SHIPPED execution path, steady-state decode.

Live server (GPU 0) exposes no /start_profile in this build, so per §21 we
capture offline on GPU 1 with every prod serve arg mirrored — including
VLLM_USE_BREAKABLE_CUDAGRAPH=1 (profile what ships, NOT enforce_eager).
Supervisor schedule(wait=2, warmup=2, active=5, repeat=1) intends "capture
steady state, skip cold start"; offline LLM() has no per-engine-step hook to
drive profiler.step(), so the equivalent is: unprofiled warmup generates,
then full-window capture of the 256-token temp-0 decode.
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
                w.writerow([e.key, e.count, e.cuda_time_total, e.cpu_time_total])

    prof.export_chrome_trace("/qwork/prof_r12_trace.json")
    print("TRACE_EXPORTED /qwork/prof_r12_trace.json", flush=True)


if __name__ == "__main__":
    main()
