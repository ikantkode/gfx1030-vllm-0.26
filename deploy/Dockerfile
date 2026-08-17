# Patched vLLM image for gfx1030 (Radeon PRO V620 / RDNA2) — all kernels baked in.
# Thin layer over the prebuilt vLLM-RDNA image: four file copies, no source rebuild.
#
# Build from the repo root:
#   docker build -f deploy/Dockerfile -t ghcr.io/ikantkode/gfx1030-vllm-0.26:local .

FROM blivioniag/vllm-rdna:v0.26.0

COPY patches/awq_triton.py /src/vllm/vllm/model_executor/layers/quantization/awq_triton.py
COPY patches/utils.py /src/vllm/vllm/model_executor/layers/utils.py
COPY patches/rmsnorm_gfx1030.py /src/vllm/vllm/model_executor/layers/rmsnorm_gfx1030.py
COPY patches/chunked_prefill_paged_decode.py /src/vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
