# gfx1030-vllm-0.26 — vLLM 0.26 optimized for AMD RDNA2 (gfx1030 / Radeon PRO V620)

**One-command deploy:** see [`deploy/`](deploy/) — `./quickstart.sh` pulls the model
from Hugging Face, builds the patched image, and serves at ~98 tok/s. Downloadable
release zips under [Releases](../../releases).

Surgical kernel patches **plus one re-quantized checkpoint** that take **single-stream LLM
decode from ~10 to 97.9 tokens/s (9.8×)** on unsupported AMD gfx1030 hardware — no vLLM
rebuild, no Triton fork, all mounted as files into a prebuilt docker image (v1.1.0: the
lm_head is now INT4 in the model weights — the image itself is unchanged from v1.0.0).

Tested with: `ikantkode/Qwen3.5-4B-AWQ-vd` (v1.1.0 — INT4 lm_head; hybrid GDN linear-attention +
full-attention multimodal model, AWQ-INT4 MLP + attention + **lm_head** — our re-quant of
`Qwen/Qwen3.5-4B`, see `requant/`; distinct from `QuantTrio/Qwen3.5-4B-AWQ` used at
rungs 0–7),
2× AMD Radeon PRO V620 (gfx1030, RDNA2, 32 GB), ROCm 7.2 host driver, Ubuntu 24.04.

---

## Required docker image (no build needed)

```bash
docker pull blivioniag/vllm-rdna:v0.26.0     # ~61 GB — vLLM 0.26.1.dev built for gfx1030
```

Everything in this repo is **file-mount patches** on top of that image. You never compile anything.

## Model

Any AWQ-INT4 checkpoint works with the kernel patches; the tested one is
[`ikantkode/Qwen3.5-4B-AWQ-vd`](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd)
(~5 GB, full-INT4 incl. the lm_head — v1.1.0). The v1.0.0 checkpoint
[`ikantkode/Qwen3.5-4B-AWQ-vd`](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd)
(the v1.0.0 revision) gave 84.5 tok/s; v1.1.0 of the same repo adds the
+15.9% single-stream win.

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
| 7 | `utils.py`: Triton fp16 GEMV for n==1, k>8192 (LLMM1 can't launch there; was ~700 µs rocBLAS) | 45.2 |
| 8 | **Re-quant `self_attn` + `linear_attn` to INT4** (`requant/`; fp16 weight read 2.53 → 0.65 GB/token) | **51.7** |
| 9 | `awq_triton.py`: -vd shape-set GEMV tile re-sweep (135-config grid, L2-flushed; per-(N,K) dispatch table) | 53.7 |
| 10 | `awq_triton.py`: **K-split GEMV for M==1** (`awq_gemv_splitk_kernel` grid (N/BN, 16) + fp32 partials + reduce; INT4 block 13.3 → 9.1 ms/token cold) | **62.3** |
| 11 | `awq_triton.py`: persistent splitk partials cache (`_SPLITK_PARTIALS`, keyed (N, split, device); graph-safe) | 62.3 (neutral, kept) |
| 13 | `rmsnorm_gfx1030.py`: **fused Gemma RMSNorm** (one Triton kernel vs the 10–13-launch ATen chain the IR `rms_norm` op falls back to; ×81 norms/token; monkey-patched at import) | **74.4** |
| 14 | `chunked_prefill_paged_decode.py`: **num_warps=8 at the paged-attn Triton launch** (the stock launch passes no warps → JIT default 4 on a 4-workgroup grid across 72 CUs; latency-bound gather loop halves: 226 → 112 µs/call at seq 281) | **79.1** |
| 15 | `awq_triton.py`: **per-shape splitk config table** (true decode set = 120 calls/5 shapes; big-N K=2560 → BN=128/BK=32/SP=4/W=4) | **84.5** |
| 16/18 | `awq_triton.py`: **M>32 + M≤32 per-(N,K) tiled-GEMM tables** (multi-user knee; single-stream unchanged — see CHANGELOG) | 84.5¹ |
| 19 | **Re-quant the lm_head to INT4** (the single largest decode GEMV, 2560→248320; added in place to the `-vd` repo as v1.1.0, kernel-neutral — the image is unchanged from v1.0.0) | **97.9** |

¹ Rungs 16/18 move the 128-user knee (966.6 tok/s aggregate), not single-stream.

**Decode budget** (12.64 ms/token wall at 79.1 TPS; Entry 26/27/28):
INT4 splitk GEMV 6.21 (128 calls) + lm_head 2.89 (at fp16 byte floor) + paged_attention 0.80
(8 × ~100 µs; was 8 × 183 µs — rung 14; the in-tree `paged_attention_rocm` custom kernel is
triple-dead on this stack: arch-gated away from gfx1030, no HEAD=256 template instantiation,
non-pow2 528 block size forces the Triton fallback) + norms 0.17 (was 1.93 over ~900 launches; now 81 ×
~2.1 µs fused) + FLA 0.31 + reduce 0.19 + rest ~0.3; live-wall residual ~1.9 (launch/replay/
CPU; rung 13 removed ~850 launches/token, ~1536 → ~700). Cold INT4 block was 9.1 ms/token —
live-warm runs 6.21, i.e. no contention inflation. Rung 13 root cause: vLLM 0.26 ships fused
`_C` RMSNorm kernels but the IR provider gate wants `weight.dtype == x.dtype` (Gemma-style
norms pass an fp32 weight) and no op priority is ever configured — so every norm ran the
composite native chain; the server even logs `Priority not set for op rms_norm, using native
implementation`.

Physics ceiling after rung 8 ≈ **113–145 TPS** (weight read 5.4 → ~3.5 GB/token at ~445 GB/s
effective; was 74–95 before the attention re-quant). Byte-exact floor from checkpoint ground
truth (v1.1.0, untied INT4 lm_head): **~2.0 GB/token** (INT4 1.85 + INT4 lm_head ~0.25) →
**~175 TPS hard ceiling** (was 3.12 GB/token / ~142 with the tied fp16 head). We're at 97.9 —
the split-K decode GEMV (~5.5 ms/token) is the dominant remaining gap to the ceiling.

## Roadmap

1. ~~**Re-quantize `self_attn` + `linear_attn` to INT4**~~ — **DONE, rung 8, 51.7 TPS** (see
   `requant/`). Cuts the FP16 read 2.53 → 0.65 GB/token; ceiling → 113–145 TPS. Uses the
   [`quivent/autoawq-qwen35`](https://github.com/quivent/autoawq-qwen35) AutoAWQ fork; the patched
   GEMV kernel automatically serves every newly-quantized layer.
2a. ~~**Re-quantize the lm_head to INT4**~~ — **DONE, rung 19, 97.9 TPS (v1.1.0)**. The head is
   the single largest decode GEMV (2560→248320, 1.56 GB fp16). It was originally "not quantizable"
   (tied embeddings — vLLM never creates a quantizable lm_head module); the fix is to **untie**
   (`tie_word_embeddings: false` + `quantization_config.lm_head: true`) and ship the head under the
   top-level `lm_head.` prefix in `model_lmhead.safetensors` (the vLLM 0.26 load path). Kernel-
   neutral: the stock `awq_triton.py` serves the N=248320 K=2560 shape (SPLIT_K=1). See
   `requant/` and the INT4 lm_head added to the `ikantkode/Qwen3.5-4B-AWQ-vd` repo (v1.1.0).
   **Gotcha that cost a day:** the fork's activation-smoothing writes input-LN folds Llama-style
   (`w = s·w_base`), but Qwen3.5's RMSNorm applies gain `(1 + w)` — the delivered scale becomes
   `(1+s·w)` instead of `s·(1+w)` and generation is garbage at serve time while every per-module
   check looks clean. `requant/quant.py` post-passes fix the storage to `w = s·(1+w_base) − 1`
   (per-channel s recovered by least squares from the folded consumers). If you re-quant with the
   raw fork and output is gibberish, this is why.
2. **paged_attention decode config** (~1.46 ms/token): 8 × 183 µs for ~2.6 MB KV at batch 1 —
   ~3% of peak BW; should be 10-30 µs. Investigate the ROCm attention dispatch gating first.
3. **splitk occupancy** (~2.0 ms/token vs byte floor), `fused_qk_rmsnorm_rope` ROCm/VL gate
   flip, and the remaining elementwise/cast storm: torch.compile/inductor can freeze gfx1030
   (ROCm #5572), so each needs a hand-fused or dispatch-level fix.
4. **Multi-user phase**: batched throughput on top of this stack (not started).

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
│   ├── rmsnorm_gfx1030.py           # from patches/ in this repo (rung 13)
│   ├── chunked_prefill_paged_decode.py  # from patches/ in this repo (rung 14)
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
# TPS = completion_tokens / wall_seconds  (expect ~79 on a V620)
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
patches/    awq_triton.py(.orig/.patch), utils.py(.orig/.patch), rmsnorm_gfx1030.py,
            gates/ (test_r13*.py — offline numerics + launch-count + device-time gate)
compose/    docker-compose.yml + override (the working serving config)
benches/    live kernel sweep harnesses (AWQ gemv/dot, fp16 gemv, FLA)
profiling/  offline decode-attribution scripts
requant/    self_attn+linear_attn INT4 re-quantization pipeline (AutoAWQ + autoawq-qwen35 fork;
            produces Qwen3.5-4B-AWQ-vd from Qwen/Qwen3.5-4B, incl. the (1+w) LN-fold post-pass)
```
