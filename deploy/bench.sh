#!/usr/bin/env bash
# Speed check: tokens per second for a single stream.
set -euo pipefail
URL=${URL:-http://localhost:8000}

curl -s "$URL/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"/model","prompt":"hello","max_tokens":4,"temperature":0}' > /dev/null

python3 - "$URL" <<'EOF'
import json, sys, time, urllib.request
url = sys.argv[1]
body = json.dumps({"model": "/model",
                   "prompt": "Write a long detailed essay about the history of computing.",
                   "max_tokens": 256, "temperature": 0, "ignore_eos": True}).encode()
req = urllib.request.Request(url + "/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req))
dt = time.time() - t0
n = r["usage"]["completion_tokens"]
print("%d tokens in %.2fs = %.1f tokens/sec" % (n, dt, n / dt))
EOF
