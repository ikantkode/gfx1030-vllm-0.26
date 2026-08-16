import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

BASE = "/qwork/Qwen3.5-4B"
OUT = "/hostq/Qwen3.5-4B-AWQ-vd"

print(">>> VARIANT D: FULL quant config (self_attn + GDN qkv/z + MLP INT4; in_proj_a/b fp16)", flush=True)
print(">>> loading base model (fp16)", flush=True)
model = AutoAWQForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, safetensors=True, trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

# --- get_layers_for_scaling wrapper (supersedes the VARIANT C shim) ----------
# The fork's get_layers_for_scaling hardcodes activation features per attention
# family (input_feat["linear_attn.in_proj_qkv"] for GDN layers,
# input_feat["self_attn.q_proj"] for full-attention layers); input_feat only
# holds keys for linears actually quantized, so an excluded family -> KeyError.
# Variant D uses the fork-default mtc (only in_proj_a/b excluded), so every
# hardcoded entry resolves and the KeyError path should never fire — the
# fallback is kept as a hard guard: if it ever triggers in a FULL-config run
# something is wrong and we must not silently quantize an unsmoothed layer.
#
# MLP up/down identity-scale fix (root cause, Stage E Entry 9): the fork's
# native entries search per-branch scales for gate+up (prev_op =
# post_attention_layernorm) and for down (prev_op = up_proj). That leaves
# up_proj ~31% and down_proj ~24% functional error — nothing compensates the
# mismatched search scales. OLD pipeline (no smoothing anywhere) is proven at
# 9.4-10.3%. So we substitute ALL-ONES scale vectors for the up/down entries:
# the AWQ search sees no signal, scales stay identity, ln is untouched on
# those paths -> quantization of up/down becomes plain g128, unsmoothed,
# exactly the OLD behavior. gate_proj is left exactly as the fork builds it
# (measured 9.7%, fine).
from awq.models.qwen3_5 import Qwen3_5AWQForCausalLM

_orig_get_layers_for_scaling = Qwen3_5AWQForCausalLM.get_layers_for_scaling


def _identity_scale(entry):
    """Return a copy of a scaling entry with `inp` replaced by ones."""
    e = dict(entry)
    inp = e.get("inp")
    if isinstance(inp, (list, tuple)):
        e["inp"] = [torch.ones_like(x) for x in inp]
    else:
        e["inp"] = torch.ones_like(inp)
    return e


@staticmethod
def _get_layers_for_scaling_variant_d(module, input_feat, module_kwargs):
    entries = _orig_get_layers_for_scaling(module, input_feat, module_kwargs)
    fixed = []
    for e in entries:
        if "mlp.up_proj" in e.get("layers", []) or "mlp.down_proj" in e.get("layers", []):
            fixed.append(_identity_scale(e))
        else:
            fixed.append(e)
    print(f"    [VD] layer {getattr(module, 'layer_idx', '?')}: scaling entries "
          f"{len(entries)} -> up/down identity-scaled", flush=True)
    return fixed


Qwen3_5AWQForCausalLM.get_layers_for_scaling = _get_layers_for_scaling_variant_d
print(">>> VD scaling wrapper applied (MLP up/down get identity smoothing; "
      "KeyError guard kept from variant C)", flush=True)

model.quantize(
    tok,
    # calib memory fallback (PLAN 1.3), unchanged from attempts 6-11
    max_calib_samples=64,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
        # Variant D = fork-default mtc: quantize self_attn + GDN qkv/z/out +
        # MLP; only in_proj_a/b (+ visual/mtp) stay fp16. setup-and-quant.sh
        # step 2 seds the class attribute to exactly this list (the class
        # attribute overrides the user quant_config — awq/models/base.py:222).
        "modules_to_not_convert": ["visual", "mtp", "in_proj_b", "in_proj_a"],
    },
)

# --- in_proj_a/b fp16 compensation (Stage E root cause) ----------------------
# The fork's smoothing multiplies each layer's input_layernorm weight by s
# (s = ln_ckpt/ln_base, measured per channel) and folds 1/s into the QUANTIZED
# consumers only. in_proj_a/b are fp16 (mtc) and were never compensated ->
# their effective output error is 96-156%. The ln output is s-scaled, so the
# fp16 consumers must be divided by s on their input channels (W is [N,K]:
# column k <- W[:,k]/s_k). Mathematically exact; numerically a pure rescale of
# fp16 weights.
print(">>> VD post-pass: compensating fp16 in_proj_a/b with 1/s per input channel", flush=True)
compensated = 0
# wrapper tree (verified in-container): AutoAWQ top -> .model (ConditionalGen)
# -> .model (Qwen3_5Model) -> .language_model (Qwen3_5TextModel) -> .layers (32 blocks).
# DEVICE LAYOUT AFTER quantize() (probed): input_layernorm weights live on CPU
# (the AWQ loop moves each layer back to CPU as it goes), while the excluded
# fp16 in_proj_a/b modules stay on cuda:0. So compute s in fp32 on CPU and move
# only s to the (GPU) weight's device for the division.
text_model = model.model.model.language_model

# ln_base must be read from the pristine base safetensors on disk: after
# quantize() the in-memory input_layernorm IS the already-scaled ln_ckpt, so
# it cannot serve as the denominator. s = ln_ckpt / ln_base per channel.
def _load_base_layernorms(base_dir):
    from safetensors import safe_open
    import glob
    out = []
    for i in range(len(text_model.layers)):
        key = f"model.language_model.layers.{i}.input_layernorm.weight"
        t = None
        for f in sorted(glob.glob(f"{base_dir}/*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                if key in sf.keys():
                    t = sf.get_tensor(key)
                    break
        assert t is not None, f"base input_layernorm missing for layer {i}"
        out.append(t)
    return out

base_lns = _load_base_layernorms(BASE)
# Only GDN (linear_attention) layers carry in_proj_a/b — self_attn layers have no
# such modules and would raise AttributeError (crash 3: first self_attn layer 3).
# Guard by attribute presence so the 24 GDN layers are compensated and the 8
# self_attn layers are skipped cleanly.
for i, layer in enumerate(text_model.layers):
    gdn = getattr(layer, "linear_attn", None)
    if gdn is None:
        continue
    ln_ck = layer.input_layernorm.weight.data  # CPU fp16 post-quant
    s = (ln_ck.float().cpu() / base_lns[i].float().cpu())     # [K] fp32 on CPU
    for name in ("in_proj_a", "in_proj_b"):
        w = getattr(gdn, name).weight.data
        s_w = s.to(device=w.device, dtype=w.dtype)
        w.copy_((w.float() / s_w.unsqueeze(0)).to(w.dtype))
    compensated += 1
print(f">>> VD compensated {compensated} GDN layers (in_proj_a/b /= s)", flush=True)

# --- LN storage fix (Entry 20) -----------------------------------------------
# The fork's smoothing wrote the fold Llama-style: w_ckpt = s*w_base. But
# transformers' Qwen3_5RMSNorm applies gain (1 + weight), so the delivered
# per-channel scale is (1 + s*w), not the intended s*(1+w) — every LN output
# channel mis-scaled, token stream garbage. Correct storage:
#     w_new = s*(1 + w_base) - 1
# with the SAME per-channel s the consumers were folded with. Recover s
# robustly by least squares from a consumer's columns (fp16 in_proj_a on GDN
# layers; dequantized q_proj on self_attn layers) — the LN ratio alone is
# unstable where w_base ~= 0. See /qwork/ln_fix_checkpoint.py (the applied,
# verified equivalent on the shipped checkpoint).
def _ls_scale(wb, wv):
    return (wb * wv).sum(0) / (wv * wv).sum(0).clamp_min(1e-8)

def _dequant(sd_key_prefix, holder):
    from awq.utils.packing_utils import dequantize_gemm
    q, sc, z = (holder[f"{sd_key_prefix}.{k}"]
                for k in ("qweight", "scales", "qzeros"))
    return dequantize_gemm(q, z, sc, 4, 128).float().T  # [out, in]

print(">>> VD post-pass 2: (1+w)-correct input_layernorm storage", flush=True)
fixed = 0
for i, layer in enumerate(text_model.layers):
    key = f"model.language_model.layers.{i}.input_layernorm.weight"
    wb = base_lns[i].float()
    ln_ck = layer.input_layernorm.weight.data.float().cpu()
    if torch.allclose(wb, ln_ck, atol=1e-7):
        continue
    gdn = getattr(layer, "linear_attn", None)
    if gdn is not None:
        # fp16, exact: W_base[:,c] = s[c] * W_ckpt[:,c]  (post-compensation)
        a_key = f"model.language_model.layers.{i}.linear_attn.in_proj_a.weight"
        w_a_b = None
        for f in sorted(glob.glob(f"{BASE}/*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                if a_key in sf.keys():
                    w_a_b = sf.get_tensor(a_key).float()
                    break
        assert w_a_b is not None, f"{a_key} missing in base"
        w_a_v = gdn.in_proj_a.weight.data.float().cpu()
        s = _ls_scale(w_a_b, w_a_v)
    else:
        # dequantized q_proj columns vs base fp16
        q_key = f"model.language_model.layers.{i}.self_attn.q_proj"
        w_q_b = None
        for f in sorted(glob.glob(f"{BASE}/*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                if f"{q_key}.weight" in sf.keys():
                    w_q_b = sf.get_tensor(f"{q_key}.weight").float()
                    break
        holder = {f"{q_key}.{k}": getattr(layer.self_attn.q_proj, k)
                  for k in ("qweight", "scales", "qzeros")}
        holder = {k: v.detach().float().cpu() for k, v in holder.items()}
        s = _ls_scale(w_q_b, _dequant(q_key, holder))
    w_new = (s * (1 + wb) - 1)
    assert torch.isfinite(w_new).all(), f"L{i}: non-finite"
    layer.input_layernorm.weight.data.copy_(
        w_new.to(layer.input_layernorm.weight.dtype))
    fixed += 1
print(f">>> VD fixed {fixed} input_layernorm tensors ((1+w)-correct)", flush=True)

print(">>> saving", flush=True)
model.save_quantized(OUT)
tok.save_pretrained(OUT)
print(">>> QUANT_SAVED", flush=True)
