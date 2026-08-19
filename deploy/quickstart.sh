#!/usr/bin/env bash
# One-line setup: download the model (~5 GB, once) and start the server.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f model/config.json ]; then
  echo ">> Downloading the model (about 5 GB, one time only)..."
  huggingface-cli download ikantkode/Qwen3.5-4B-AWQ-vd-lmhead-int4 --local-dir ./model
fi

echo ">> Building/starting the server (first run takes a few minutes)..."
docker compose up -d --build

echo ">> Waiting for the model to load (~3 minutes)..."
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || true)
  [ "$code" = "200" ] && break
  sleep 10
done

if [ "$code" = "200" ]; then
  echo ""
  echo ">> Ready! Your AI server is at:  http://$(hostname -I | awk '{print $1}'):8000/v1"
  echo ">> Check the speed with:        ./bench.sh"
else
  echo ">> Still starting. Check: docker logs qwen-vllm"
fi
