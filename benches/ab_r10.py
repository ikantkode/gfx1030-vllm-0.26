import json, urllib.request

PROMPTS = [
    "Explain photosynthesis to a 10 year old.",
    "Write a Python function to reverse a linked list.",
    "Summarize the causes of World War I.",
    "What is 17*24? Show your reasoning.",
    "Translate 'the weather is nice today' to French.",
]

out = []
for p in PROMPTS:
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions",
        data=json.dumps({"model": "/model", "prompt": p,
                         "max_tokens": 128, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    out.append(d["choices"][0]["text"])
    print(f"=== {p}\n{out[-1]}\n", flush=True)

with open("/home/beefyboi1/qwen/new_outputs_r10.txt", "w") as f:
    f.write("\n".join(out))
print("saved new_outputs_r10.txt")
