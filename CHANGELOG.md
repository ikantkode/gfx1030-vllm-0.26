# CHANGELOG — TPS ladder, rungs → commits

Single-stream decode TPS for Qwen3.5-4B-AWQ-vd on one Radeon PRO V620
(gfx1030), vLLM 0.26.1.dev. Each tag/commit below is a bisect point:
`git log --oneline`, or `git tag` for the TPS milestones.
Live numbers = bench ×3 on the serving container (PROGRESS.md has the full
per-rung detail, offline gates, and A/B fidelity evidence).

| Rung | TPS | Change | Commit / tag |
|---|---|---|---|
| 0 | ~10 | Baseline (AWQ fp16 paths, eager) | `fda1aa3` (rungs 0–7 bundled) |
| 1 | 12.0 | `SPLIT_K=1` for M≤32 decode in `awq_triton.py` | 〃 |
| 2 | 16.8 | Breakable CUDA graphs (no `--enforce-eager`, no torch.compile) | 〃 |
| 3 | 26.3 | Unlock AMD **LLMM1** skinny-decode kernel for gfx1030 (arch gate in `utils.py`) | 〃 |
| 4 | 34.5 | Swept GEMM tiles `BM=16/BN=128/BK=64/W=8/S=3` | 〃 |
| 5 | 38.9 | Shape-aware `SPLIT_K` (K≤4096→1, K>4096→8) | 〃 |
| 6 | 44.0 | Custom Triton GEMV for M==1 (no `tl.dot` M-tile waste) | 〃 |
| 7 | 45.2 | fp16 Triton GEMV for n==1,k>8192 (LLMM1 can't launch there) | 〃 |
| 8 | 51.7 | Full-architecture INT4 re-quant `-vd` (self_attn + linear_attn; LN-fold storage fix) | `1871e52` · tag `rung-8-51.7tps` |
| 9 | 53.7 | -vd shape-set tile re-sweep, per-(N,K) dispatch table (superseded by 10) | `1febbcb` · tag `rung-10-62.3tps` (rungs 9–10 one commit) |
| 10 | **62.3** | **K-split GEMV for M==1** (grid (N/64, 16), fp32 partials + reduce; one config wins all shapes) | 〃 |
| 11 | 62.3 | Persistent splitk partials cache (TPS-neutral, kept: strictly less work) | `968fc68` |
| 12 | — | Topology review only, no code (decode fully attributed, Entry 26) | `afe0c8f`/`34623ce` (profiling + budget) |
| 13 | 74.4 | **Fused Gemma RMSNorm** (Triton, 1 kernel vs 10–13-launch native chain ×81/token) | `9d4f668` · tag `rung-13-74.4tps` |
| 14 | **79.1** | **Paged-attn Triton `num_warps=8`** (grid of 4 workgroups on 72 CUs was latency-bound; 226→112 µs/call) | `5f01e3c` · tag `rung-14-79.1tps` |
| 15 | **84.5** | **Per-shape splitk config table** (true decode set = 120 calls/5 shapes, not the stale 6-shape table; big-N K=2560 → BN=128/BK=32/SP=4/W=4, 64 B/row contiguous reads). Win is a live L2/launch/occupancy effect, NOT a per-shape cold-kernel win (that protocol is floor-dominated ~190 µs, can't resolve it). A/B 5/5 byte-identical. | `df4a71b` · tag `rung-15-84.5tps` |
| 16 | **84.5**¹ | **M>32 tiled GEMM per-(N,K) table** (the multi-user "knee" branch, left at stock BM32/BN32/BK32 W4 S3). Focused M=128 sweep, 144-config grid, correctness-gated vs dequant: ONE tile (64,128,32,8,2) wins all five shapes, only split-K varies (18432→sk1, the two mid K=2560→sk4, the two K≥4096→sk8). 2.02× on the per-step AWQ block (57→115 GB/s). Single-stream TPS unchanged — the win is on the 16+ seq knee. | `this` · see v1.0.0 |
| 18 | **84.5**¹ | **M≤32 per-(N,K)×M-band table** (decode M∈{8,16,32}; ceiling band-selection). M=8/16/32 sweeps on the 5 -vd shapes: per-shape winners keyed by (N,K) with M-band ceiling (M=10 → the (16,)-band config, the M≤32 tiled GEMM at M=10). Closes the last untuned branch. Single-stream TPS unchanged. | `this` · see v1.0.0 |

¹ Rungs 16+18 are **multi-user** wins — the mns=128 (128-concurrent) knee, not the
single-stream mns=1 ladder, which stays at 84.5. Measured on the in-process
b128 serve (graphs on, GPU 1, never the live server): mns=16 433.6, mns=32 642.7,
mns=64 862.4, **mns=128 966.6** (+43.6% vs the stock-M>32 baseline 673.3 at mns=128).
Single-stream decode stays 84.5 tok/s (bench ×3: 84.7 / 84.5, run 1 = 45.0 graph-capture
warmup). Full 4-gate set PASS: offline numerics (A1 15/15 max rel 0.0186, A2 5/5 max rel
0.0171), single-stream bench, knee ladder, 5-prompt A/B coherent (Entry 38–39).

Decode wall: 100+ ms → **12.64 ms/token** at rung 14 → **~11.9 ms/token** at rung 15 (8.4×).
Hard ceiling ≈ 142 TPS (3.12 GB/token @ ~445 GB/s); realistic ceiling 85–95.

## v1.0.0
First full release: rungs 0–18 in one stack. `patches/awq_triton.py` now carries all
three dispatch tables (M==1 Rung-15 GEMV/splitk · M≤32 Rung-18 per-(N,K)×M-band · M>32
Rung-16 per-(N,K)). `gates/sweep_m128_r16.py` and `gates/sweep_m16_r18.py` are the
correctness-gated sweeps that produced the Rung-16/18 tables. Serve config =
`compose/docker-compose.override.yml` (TP=1, mns=2, breakable CUDA graphs, ROCM_ATTN).

## Layout
- `patches/` — everything mounted into the serving container by `compose/docker-compose.override.yml`
- `gates/` — offline numerical gates + probes (run on GPU 1, never the live server)
- `benches/` — live TPS bench + 5-prompt semantic A/B scripts
- `profiling/` — capture recipe (Entry 26) + analyzers
- `requant/` — tier-3 INT4 pipeline (quant.py, setup script, VD post-pass)
