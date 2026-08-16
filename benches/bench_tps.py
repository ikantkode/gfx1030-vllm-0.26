import json, sys, urllib.request

def run():
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions",
        data=json.dumps({
            "model": "/model",
            "prompt": "Write a long detailed essay about the history of computing.",
            "max_tokens": 256, "temperature": 0, "ignore_eos": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    u = d["usage"]
    print(f"{u['completion_tokens']} tokens, {u.get('total_tokens','-')} total")

# warmup
for _ in range(1):
    w = urllib.request.Request(
        "http://localhost:8000/v1/completions",
        data=json.dumps({"model": "/model", "prompt": "hello",
                         "max_tokens": 4, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(w, timeout=60).read()

import time
for i in range(3):
    t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions",
        data=json.dumps({
            "model": "/model",
            "prompt": "Write a long detailed essay about the history of computing.",
            "max_tokens": 256, "temperature": 0, "ignore_eos": True,
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    dt = time.time() - t0
    n = d["usage"]["completion_tokens"]
    print(f"run{i+1}: {n} tokens in {dt:.2f}s = {n/dt:.1f} TPS")
