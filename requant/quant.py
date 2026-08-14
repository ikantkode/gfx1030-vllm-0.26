import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

BASE = "/work/Qwen3.5-4B"
OUT = "/hostq/Qwen3.5-4B-AWQ-full"

print(">>> loading base model (fp16)", flush=True)
model = AutoAWQForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, safetensors=True, trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
print(">>> quantizing self_attn + linear_attn to INT4 (GEMM, g128)", flush=True)
model.quantize(
    tok,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
        "modules_to_not_convert": [
            "visual", "mtp", "in_proj_a", "in_proj_b", "model.layers.0."
        ],
    },
)
print(">>> saving", flush=True)
model.save_quantized(OUT)
tok.save_pretrained(OUT)
print(">>> QUANT_SAVED", flush=True)
