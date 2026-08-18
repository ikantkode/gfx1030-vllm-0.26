# Release v1.0.0 — notes for the GitHub Release

**Title:** v1.0.0 — 84.5 tok/s Qwen3.5-4B on Radeon PRO V620 (gfx1030)

**Body:**

---

## What this is

A complete, working stack to run **Qwen 3.5 4B at ~84 tokens/second on an AMD Radeon
PRO V620** — hardware vLLM doesn't officially support well. **9× faster than the
stock setup**, quality-verified at every step.

## The one-liner

Download this release (zip/tar below), then:

```bash
cd gfx1030-vllm-0.26*/deploy && ./quickstart.sh
```

Downloads the model, builds the patched image, starts the server. Point any
OpenAI-compatible app at `http://<machine>:8000/v1`.

## What's inside

- **18 measured optimization rungs** (10 → 84.5 tok/s single-stream, 966.6 tok/s at the 128-user knee): custom AWQ INT4 GEMV kernels,
  AMD LLMM1 unlock for gfx1030, K-split decode, fused RMSNorm, tuned paged attention —
  see `CHANGELOG.md` for every step with its commit
- **Deploy kit** (`deploy/`): Dockerfile (patches baked into `blivioniag/vllm-rdna:v0.26.0`
  — no vLLM rebuild), docker-compose, one-command quickstart
- **Registry image**: `ghcr.io/ikantkode/gfx1030-vllm-0.26:latest` (built by CI)
- **Model**: [`ikantkode/Qwen3.5-4B-AWQ-vd`](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd)
  — our full-INT4 re-quant (attention + GDN + MLP), 3.8 GB, works on any vLLM AWQ stack
- **Quantization recipe**: [`Qwen3.5-Quant-Recipe`](https://github.com/ikantkode/awq-quant-recipe)
  — the reproducible pipeline incl. the ALS scale-fitter and the two upstream bug
  post-mortems (the `(1+w)` norm-fold defect in the AutoAWQ fork)

## Measured

| | stock | this release |
|---|---|---|
| Single-stream decode | ~10 tok/s | **84.5 tok/s** |
| Multi-user knee (128 concurrent) | — | **966.6 tok/s** aggregate (+43.6% over stock tiles) |
| Model size (vs reference AWQ) | 5.7 GB | 3.8 GB |

## Requirements

V620/RX6800-class 32 GB GPU · ROCm · Docker. Quality gates: per-module ≤ 0.11 rel-err
vs base weights; 5-prompt A/B equivalent to the reference checkpoint.
