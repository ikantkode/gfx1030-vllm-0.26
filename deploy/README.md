# Quick Start — run the fast AI server

This folder turns any machine with an **AMD Radeon PRO V620** (or RX 6800/6900-class)
graphics card into a fast AI text server running Qwen 3.5 4B at **~84 tokens per
second** — about 9× faster than the normal setup on this card.

No coding needed. One command does everything:

```bash
./quickstart.sh
```

That downloads the model (about 4 GB, one time), builds the server, and starts it.
When it finishes it prints your server address — looks like `http://192.168.x.x:8000/v1`.
Point any OpenAI-compatible chat app at it, or test from a terminal:

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"/model","prompt":"Explain photosynthesis to a 10 year old.","max_tokens":200}'
```

## What you need

- An AMD V620 / RX 6800-class GPU (32 GB), ROCm installed and working, Docker
- ~25 GB free disk · ~3 minutes of patience after each start

## Everyday commands

| What you want | Command |
|---|---|
| Start | `docker compose up -d` |
| Stop | `docker compose down` |
| Health check (want: 200) | `curl http://localhost:8000/health` |
| Watch logs | `docker logs -f qwen-vllm` |
| Speed check | `./bench.sh` |

## More people at once (optional)

By default: 2 simultaneous users, 8,000-word memory, fastest replies. For 5 users with
32,000-word memories, change in `docker-compose.yml`: `"8192"`→`"32768"`,
`"2"`→`"8"`, `"2048"`→`"4096"`, then `docker compose up -d --force-recreate`.
That yields ~147 tokens/sec combined (~29 per person).

## If something goes wrong

- Not healthy after 5 min → `docker logs qwen-vllm` (last lines name the problem)
- "no space" during first build → free disk, the base server needs ~20 GB
- Don't change other settings — they're hand-tuned; several prevent known AMD crashes

## For developers

`Dockerfile` bakes the four tuned kernels from `../patches/` into the standard
`blivioniag/vllm-rdna:v0.26.0` image (no vLLM rebuild). CI publishes the image to
`ghcr.io/ikantkode/gfx1030-vllm-0.26` on every push. Model:
`ikantkode/Qwen3.5-4B-AWQ-vd`. Full technical history: repo root README + CHANGELOG.
