---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- awq
- int4
- quantization
- amd
- gfx1030
- vllm
---

# Qwen3.5-4B-AWQ-vd (v1.1.0 — INT4 lm_head)

An independently re-quantized **AWQ INT4** build of Qwen/Qwen3.5-4B with **all linear
projections quantized** — self-attention (8 full-attention layers), the gated
delta-net / linear-attention projections (24 layers), the MLPs, **and the lm_head** —
while keeping `in_proj_a`/`in_proj_b`, norms, the input embeddings and the MTP head in
fp16.

**v1.1.0 is a revision of this same repo** (no new checkpoint id — re-pull
`ikantkode/Qwen3.5-4B-AWQ-vd`). It is the v1.0.0 `-vd` checkpoint
plus one change: the **lm_head** (the 2560 → 248320 vocab projection, the single largest
decode GEMV, a 1.56 GB fp16 read every token) is now re-quantized to the same AWQ INT4
as the rest. The head is **untied** — `tie_word_embeddings: false` (top-level and in
`text_config`) with `quantization_config.lm_head: true` — so the weights live under the
top-level `lm_head.` prefix in `model_lmhead.safetensors`, exactly the load path vLLM
0.26 expects.

**~24% fewer weight bytes per token than the reference** `QuantTrio/Qwen3.5-4B-AWQ`
(~2.0 GB/token vs ~2.6), with output quality verified equivalent in side-by-side A/B
testing.

## Why the lm_head

At decode the lm_head is the largest single GEMV (vocab 248320 × hidden 2560). In fp16 it
is a 1.56 GB read per token — the single biggest weight-traffic item after the AWQ layer
weights. Re-quantizing it to INT4 cuts that read from ~2.9 ms/token to ~0.7 ms/token
(−2.2 ms, ~1.1 GB less read per token), which is the v1.1.0 throughput win.

## Measured results

Radeon PRO V620 (gfx1030), vLLM 0.26 + the companion gfx1030 kernel patches, graphs on.

| | QuantTrio/Qwen3.5-4B-AWQ (reference) | v1.0.0 `-vd` (fp16 head) | **this model (v1.1.0)** |
|---|---|---|---|
| Size | 5.7 GB | 3.8 GB | **~5.0 GB**¹ |
| Single-stream decode | 45.5 tok/s | 84.5 tok/s | **97.9 tok/s** (+15.9%) |
| 16 concurrent users (aggregate) | — | 433.6 tok/s | **553.1 tok/s** (+27.6%) |
| 64 concurrent users (aggregate) | — | 862.4 tok/s | **920.4 tok/s** (+6.7%) |
| 128 concurrent users (peak) | — | 966.6 tok/s | 955.8 tok/s (unchanged²) |

¹ `model.safetensors` (3.80 GB, the -vd body) + `model_lmhead.safetensors` (1.60 GB, the
INT4 head) + MTP head (0.24 GB) + tokenizer. The fp16 lm_head is gone — it is now INT4
inside `model_lmhead.safetensors`.
² The multi-user gain is front-loaded: the head saves a fixed ~2.2 ms/token, a large share
of the short low-batch step but a sliver of the weight-bandwidth-bound high-batch step
(full AWQ weight matrix ≈ 1.7 GB/token). Mid-range concurrency (≤~64 users) gains
+7–28%; the 128-user peak aggregate does not move (single-batch noise).

Quality gates: every module (incl. the head) ≤ 0.11 relative error vs base weights
(stock-dequant reference); full 4-gate set PASS (offline numerics, single-stream bench,
knee ladder, 5-prompt A/B).

The throughput figures require the companion gfx1030 kernel patches
(`ikantkode/gfx1030-vllm-0.26` on GitHub). On other hardware this checkpoint still
benefits any vLLM AWQ path via the reduced weight traffic. **Note:** the v1.1.0 Docker
image is unchanged from v1.0.0 — the win is entirely in these weights.

## Usage

```bash
vllm serve ikantkode/Qwen3.5-4B-AWQ-vd --dtype float16 --max-model-len 8192
```

Or one-command on a Radeon PRO V620 / gfx1030:
`ikantkode/Qwen3.5-vLLM-Deploy` (Docker, ~98 tok/s out of the box).

## Files & lineage

- Base: `Qwen/Qwen3.5-4B` (Apache-2.0)
- Quantization: AWQ INT4, group_size 128, asymmetric zero-point, GEMM packing;
  per-group scales/zeros fitted by alternating least-squares; no LN smoothing
- `model.safetensors`: the -vd body (attention + GDN + MLP INT4; in_proj/norms/embeddings
  fp16)
- `model_lmhead.safetensors`: the untied INT4 lm_head (top-level `lm_head.` prefix)
- `model_mtp.safetensors`: MTP head for speculative decoding (`qwen3_next_mtp`).
  Note: on gfx1030 with heavily M=1-optimized decode kernels, MTP verification was
  measurably slower than plain decoding; on other stacks it may help.
- Predecessor: the v1.0.0 revision of this same repo
  ([`ikantkode/Qwen3.5-4B-AWQ-vd`](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd);
  fp16 lm_head, 84.5 tok/s). v1.1.0 adds `model_lmhead.safetensors` in place.
- Full technical trail: `ikantkode/gfx1030-vllm-0.26` (README + PROGRESS log)
