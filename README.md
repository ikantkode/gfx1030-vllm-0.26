# gfx1030-vllm-0.26 — vLLM 0.26 optimized for AMD RDNA2 (gfx1030 / Radeon PRO V620)

Surgical kernel patches that take **single-stream LLM decode from ~10 to 45.2 tokens/s (4.5×)** on
unsupported AMD gfx1030 hardware — no vLLM rebuild, no Triton fork, all mounted as files into a
prebuilt docker image.

Tested with: `Qwen3.5-4B-AWQ` (hybrid GDN linear-attention + full-attention multimodal model, AWQ-INT4 MLP),
2× AMD Radeon PRO V620 (gfx1030, RDNA2, 32 GB), ROCm 7.2 host driver, Ubuntu 24.04.

---

## Required docker image (no build needed)

```bash
docker pull blivioniag/vllm-rdna:v0.26.0     # ~61 GB — vLLM 0.26.1.dev built for gfx1030
```

Everything in this repo is **file-mount patches** on top of that image. You never compile anything.

## Model

Any AWQ-INT4 checkpoint works with the kernel patches; the tested one is
[`QuantTrio/Qwen3.5-4B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.5-4B-AWQ) (~6 GB).

---

## Where we started → where we are

**Starting problems (Aug 13, 2026):**
1. vLLM refused/failed on gfx1030 — every accelerated path is arch-gated to `gfx9`/`gfx11+`; gfx1030 matches nothing.
2. `--attention-backend TRITON_ATTN` crashed: `OutOfResources: shared memory, required 139264, limit 65536` (gfx1030 per-workgroup LDS cap).
3. Default runtime config produced **negative KV cache** (`Available KV cache memory: -10.61 GiB`) and aborted.
4. Where it ran at all: **~10 TPS** single-stream — kernels at ~5-10% of memory bandwidth.

**Result ladder (each step live-measured, 256-token completion, temp 0):**

| # | Change | TPS |
|---|---|---|
| 0 | Baseline (image defaults, correct config) | ~10 |
| 1 | `awq_triton.py`: `SPLIT_K=1` for M≤32 decode (drop 8-way partial+reduce) | 12.0 |
| 2 | Drop `--enforce-eager` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (graphs w/o torch.compile) | 16.8 |
| 3 | `utils.py`: unlock AMD's **LLMM1** skinny-decode kernel for gfx1030 (one arch-gate line; 3.6-5.6× vs rocBLAS at n==1) | 26.3 |
| 4 | `awq_triton.py`: swept tiles `16/128/64/W8/S3` | 34.5 |
| 5 | `awq_triton.py`: shape-aware `SPLIT_K` (K≤4096→1, K>4096→8) | 38.9 |
| 6 | `awq_triton.py`: **custom GEMV kernel for M==1** (no `tl.dot` M-tile waste) | 44.0 |
| 7 | `utils.py`: Triton fp16 GEMV for n==1, k>8192 (LLMM1 can't launch there; was ~700 µs rocBLAS) | **45.2** |

**Decode budget today (~22 ms/token at 45.2 TPS), from a real torch.profiler capture:**

| Component | ms/token | Status |
|---|---|---|
| LLMM1 fp16 GEMVs (the 4.26 GB FP16 weight read) | 9.4 | at bandwidth — only re-quantization fixes this |
| AWQ INT4 GEMV (custom kernel) | 6.1 | ~2× above floor |
| elementwise / dtype-casts / copies (~1,700 tiny ops/token) | 4.2 | unfused (needs torch.compile-class machinery; freeze-risk on gfx1030) |
| paged attention (ROCM_ATTN) | 0.8 | fine |
| FLA/GDN recurrent decode | 0.3 | optimal (swept; stock config already best) |
| fp16 GEMV k>8192 | ~0.1 | fixed (was 0.7) |

Physics ceiling for this model ≈ 74–95 TPS (5.4 GB/token weight read at ~445 GB/s effective).

## Roadmap

1. **Re-quantize `self_attn` + `linear_attn` to INT4** (in progress — see `requant/`): cuts the FP16
   read 2.53 → 0.65 GB/token, ceiling → 113–145 TPS, realistic served **~55–65 TPS**. Uses the
   [`quivent/autoawq-qwen35`](https://github.com/quivent/autoawq-qwen35) AutoAWQ fork; the patched
   GEMV kernel automatically serves every newly-quantized layer. lm_head is **not** quantizable
   (tied embeddings — vLLM never creates a quantizable lm_head module).
2. **Elementwise/cast storm** (~4.2 ms/token): spread across 378 casts + 430 copies per token; no
   single cut. Needs fusion machinery; torch.compile/inductor can freeze gfx1030 (ROCm #5572), so
   this is parked unless a safe path appears.
3. **Multi-user phase**: batched throughput on top of this stack (not started).

---

## Replication from scratch

### 1. Prerequisites
Host with ROCm kernel driver (`/dev/kfd`, `/dev/dri` present), your user in `render`+`video`
groups, docker + compose plugin.

### 2. Layout
```
~/qwen/
├── Qwen3.5-4B-AWQ/                  # HF checkpoint
├── gfx1030-patches/
│   ├── awq_triton.py                # from patches/ in this repo
│   ├── utils.py                     # from patches/ in this repo
│   └── .triton/                     # (optional, created on first run) persistent JIT cache
├── docker-compose.yml               # from compose/
└── docker-compose.override.yml      # from compose/ — mounts the patches + cache
```

### 3. Start
```bash
cd ~/qwen
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
# ready in ~10-14 min on FIRST start (one-time Triton JIT), ~2-3 min after (cache mounted)
curl http://localhost:8000/health     # -> 200
```

### 4. Benchmark (the exact measurement used for the ladder)
```bash
# warmup (first inference pays one-time kernel JIT)
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","prompt":"hello","max_tokens":4,"temperature":0}' > /dev/null
# measured run
time curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/model","prompt":"Write a long detailed essay about the history of computing.","max_tokens":256,"temperature":0,"ignore_eos":true}'
# TPS = completion_tokens / wall_seconds  (expect ~45 on a V620)
```

### 5. Applying the patches to a different image/version
`patches/*.patch` are unified diffs vs the image's vLLM tree. To apply by hand:
```bash
docker cp <container>:/src/vllm/vllm/model_executor/layers/quantization/awq_triton.py .
patch < patches/awq_triton.patch     # or copy patches/awq_triton.py wholesale
docker cp awq_triton.py <container>:/src/vllm/vllm/model_executor/layers/quantization/awq_triton.py
# same for utils.py -> vllm/model_executor/layers/utils.py
```
Full patched files are included (`patches/awq_triton.py`, `patches/utils.py`) with `.orig` baselines.

### 6. Kernel micro-benchmarks (zero downtime, run against the live server)
```bash
docker cp benches/bench_awq_gemv.py qwen-vllm:/tmp/bench.py
docker exec qwen-vllm python3 -u /tmp/bench.py     # sweeps + correctness vs reference
```
`profiling/prof2.py` = full decode attribution via torch.profiler (the tool that produced the
budget table above).

---

## Landmines (do NOT revert these)

- `--attention-backend ROCM_ATTN` — `TRITON_ATTN` exceeds gfx1030's per-workgroup LDS cap and crashes.
- Never drop `--enforce-eager` without `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (torch.compile/inductor can hard-freeze gfx1030, ROCm #5572).
- `wvSplitK` device-asserts on gfx1030 — its branch is disabled in our `utils.py`; only `LLMM1` (n==1) and the Triton GEMVs are enabled.
- `VLLM_ROCM_USE_AITER=0` — AITER does not build for gfx1030.
- `--dtype float16` — RDNA2 has no native BF16 (silent fp32 fallback = slow).
- Keep the runtime footprint small (`max-model-len 8192`, `max-num-batched-tokens 2048`,
  `mm-processor-kwargs max_pixels 1003520`) or KV cache goes negative and vLLM aborts.
- **After editing any mounted patch file: use `docker compose ... up -d --force-recreate`, never a
  plain restart** — editors that replace the inode leave the container bind-mount pointing at the
  old file. Verify: `docker exec qwen-vllm grep <marker> <file>`.

## Lessons learned (gfx1030-specific)

- Microbenchmark loops flatter results via L2 reuse — a kernel that benches at 30 µs can be 100 µs
  on cold VRAM in real decode. Trust end-to-end / profiler numbers.
- rocprof v1 cannot see the decode window on gfx1030 (and crashes on unconstrained multimodal
  profiling — pass `limit_mm_per_prompt`/`max_pixels` to offline runs or it tries to allocate
  256 GiB). Use torch.profiler instead (`profiling/prof2.py`).
- AMD ships excellent kernels inside vLLM that simply aren't gated on for this arch
  (`LLMM1`, `rocm_unquantized_gemm_impl`) — flipping arch gates is the highest-ROI move on
  unsupported GPUs.

## Repo layout
```
patches/    awq_triton.py(.orig/.patch), utils.py(.orig/.patch)
compose/    docker-compose.yml + override (the working serving config)
benches/    live kernel sweep harnesses (AWQ gemv/dot, fp16 gemv, FLA)
profiling/  offline decode-attribution scripts
requant/    WIP: self_attn+linear_attn INT4 re-quantization (AutoAWQ + autoawq-qwen35 fork)
```
