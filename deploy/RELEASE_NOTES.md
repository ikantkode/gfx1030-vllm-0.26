# Release v1.1.0 — notes for the GitHub Release

**Title:** v1.1.0 — 97.9 tok/s Qwen3.5-4B on Radeon PRO V620 (gfx1030) — INT4 lm_head

**Body:**

---

## What changed since v1.0.0

One new checkpoint. The **lm_head** — the single largest GEMV in the decode step, a
2560 → 248320 (vocab) projection that was still in fp16 (a 1.56 GB read every token) —
is now re-quantized to the same **AWQ INT4** as the rest of the model.

**Single-stream decode: 84.5 → 97.9 tok/s (+15.9%).** The head read drops
2.9 → 0.7 ms/token (−2.2 ms, ~1.1 GB less read per token).

## ⚠️ The Docker image is UNCHANGED — do not rebuild

Everything on the kernel side is **byte-identical to v1.0.0** — same
`blivioniag/vllm-rdna:v0.26.0` base, same ghcr image, same mounted patches, same Triton
cache. The win is entirely in the **model weights**. Point your existing server at the
new checkpoint and you're done:

```bash
huggingface-cli download ikantkode/Qwen3.5-4B-AWQ-vd-lmhead-int4 --local-dir ./model
docker compose up -d          # or just re-run your serve with the new path
```

No rebuild, no patch change. If you rebuilt the image for this release you wasted time.

## What's inside

- **New checkpoint**: [`ikantkode/Qwen3.5-4B-AWQ-vd-lmhead-int4`](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd-lmhead-int4)
  — the full v1.0.0 `-vd` re-quant **plus** the lm_head in AWQ INT4. The head is
  untied (`tie_word_embeddings: false` + `quantization_config.lm_head: true`) with its
  weights under the top-level `lm_head.` prefix in `model_lmhead.safetensors` — exactly
  the load path vLLM 0.26 expects.
- **Kernel-neutral**: the stock `awq_triton.py` already handles the head shape
  (N=248320, K=2560 → SPLIT_K=1; the M>32 band auto-extends). No new table entry.
- **Everything else from v1.0.0** (rungs 0–18, deploy kit, ghcr image, quant recipe) is
  unchanged.

## Measured

| | v1.0.0 | v1.1.0 |
|---|---|---|
| Single-stream decode | 84.5 tok/s | **97.9 tok/s** (+15.9%) |
| Multi-user, 16 concurrent | 433.6 tok/s | **553.1 tok/s** (+27.6%) |
| Multi-user, 32 concurrent | 642.7 tok/s | 728.7 tok/s (+13.4%) |
| Multi-user, 64 concurrent | 862.4 tok/s | 920.4 tok/s (+6.7%) |
| Multi-user, 128 concurrent (peak) | 966.6 tok/s | 955.8 tok/s (**unchanged** — noise) |

The multi-user gain is **front-loaded**: the head saves a fixed ~2.2 ms/token, a large
fraction of the short low-batch step but a sliver of the weight-bandwidth-bound
high-batch step (the full AWQ weight matrix is ~1.7 GB/token). Mid-range concurrency
(≤~64 users) gains +7–28%; the 128-user peak aggregate does not move.

## Requirements

Unchanged from v1.0.0 — V620/RX6800-class 32 GB GPU · ROCm · Docker. Quality gates:
head module ≤ 0.11 relative error vs base weights; full 4-gate set (offline numerics,
single-stream bench, knee ladder, 5-prompt A/B) PASS.

## What's next (not in this release)

Phase 4 — GEMV polish + glue fusion, profile-first. The split-K decode GEMV (~5.5
ms/token) is now the biggest single remaining cost; the head is down to ~0.7 ms.
